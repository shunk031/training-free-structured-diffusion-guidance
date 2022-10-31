import inspect
import logging
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import stanza
import torch
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionPipeline,
    StableDiffusionPipelineOutput,
    StableDiffusionSafetyChecker,
)
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from nltk.tree import Tree
from tfsdg.utils.attention_utils import KeyValueTensors
from tfsdg.utils.pipeline_utils import STRUCT_ATTENTION_TYPE
from tfsdg.utils.replace_layer import replace_cross_attention
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from transformers.tokenization_utils import BatchEncoding

logger = logging.getLogger(__name__)


@dataclass
class Span(object):
    left: int
    right: int


@dataclass
class SubNP(object):
    text: str
    span: Span


@dataclass
class AllNPs(object):
    nps: List[str]
    spans: List[Span]
    lowest_nps: List[SubNP]


class TFSDGPipeline(StableDiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
    ) -> None:
        super().__init__(
            vae,
            text_encoder,
            tokenizer,
            unet,
            scheduler,
            safety_checker,
            feature_extractor,
        )

        # replace cross attention to structured cross attention
        replace_cross_attention(nn_module=self.unet, name="unet")

        self.nlp = stanza.Pipeline(
            lang="en", processors="tokenize,pos,constituency", use_gpu=False
        )

    def preprocess_prompt(self, prompt: str) -> str:
        return prompt.lower().strip().strip(".").strip()

    def get_sub_nps(self, tree: Tree, left: int, right: int) -> List[SubNP]:

        if isinstance(tree, str) or len(tree.leaves()) == 1:
            return []

        sub_nps: List[SubNP] = []

        n_leaves = len(tree.leaves())
        n_subtree_leaves = [len(t.leaves()) for t in tree]
        offset = np.cumsum([0] + n_subtree_leaves)[: len(n_subtree_leaves)]
        assert right - left == n_leaves

        if tree.label() == "NP" and n_leaves > 1:
            sub_np = SubNP(
                text=" ".join(tree.leaves()),
                span=Span(left=int(left), right=int(right)),
            )
            sub_nps.append(sub_np)

        for i, subtree in enumerate(tree):
            sub_nps += self.get_sub_nps(
                subtree,
                left=left + offset[i],
                right=left + offset[i] + n_subtree_leaves[i],
            )
        return sub_nps

    def get_all_nps(self, tree: Tree, full_sent: Optional[str] = None) -> AllNPs:
        start = 0
        end = len(tree.leaves())

        all_sub_nps = self.get_sub_nps(tree, left=start, right=end)

        lowest_nps = []
        for i in range(len(all_sub_nps)):
            span = all_sub_nps[i].span
            lowest = True
            for j in range(len(all_sub_nps)):
                span2 = all_sub_nps[j].span
                if span2.left >= span.left and span2.right <= span.right:
                    lowest = False
                    break
            if lowest:
                lowest_nps.append(all_sub_nps[i])

        all_nps = [all_sub_np.text for all_sub_np in all_sub_nps]
        spans = [all_sub_np.span for all_sub_np in all_sub_nps]

        if full_sent and full_sent not in all_nps:
            all_nps = [full_sent] + all_nps
            spans = [Span(left=start, right=end)] + spans

        return AllNPs(nps=all_nps, spans=spans, lowest_nps=lowest_nps)

    def tokenize(self, prompt: Union[str, List[str]]) -> BatchEncoding:
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return text_input

    def _extend_string(self, nps: List[str]) -> List[str]:
        extend_nps: List[str] = []
        for i in range(len(nps)):
            if i == 0:
                extend_nps.append(nps[i])
            else:
                np = (" " + nps[i]) * (
                    self.tokenizer.model_max_length // len(nps[i].split())
                )
                extend_nps.append(np)
        return extend_nps

    def _expand_sequence(
        self, seq: torch.Tensor, length: int, dim: int = 1
    ) -> torch.Tensor:

        # shape: (77, 768) -> (768, 77)
        seq = seq.transpose(0, dim)

        max_length = seq.size(0)
        n_repeat = (max_length - 2) // length

        # shape: (10, 1)
        repeat_size = (n_repeat,) + (1,) * (len(seq.size()) - 1)

        # shape: (77,)
        eos = seq[length + 1, ...].clone()

        # shape: (750, 77)
        segment = seq[1 : length + 1, ...].repeat(*repeat_size)

        seq[1 : len(segment) + 1] = segment

        # To avoid the following error, we need to use `torch.no_grad` function:
        # RuntimeError: Output 0 of SliceBackward0 is a view and
        # # is being modified inplace. This view is the output
        # of a function that returns multiple views.
        # Such functions do not allow the output views to be modified inplace.
        # You should replace the inplace operation by an out-of-place one.
        seq[len(segment) + 1] = eos

        # shape: (768, 77) -> (77, 768)
        return seq.transpose(0, dim)

    def _align_sequence(
        self,
        full_seq: torch.Tensor,
        seq: torch.Tensor,
        span: Span,
        eos_loc: int,
        dim: int = 1,
        zero_out: bool = False,
        replace_pad: bool = False,
    ) -> torch.Tensor:

        # shape: (77, 768) -> (768, 77)
        seq = seq.transpose(0, dim)

        # shape: (77, 768) -> (768, 77)
        full_seq = full_seq.transpose(0, dim)

        start, end = span.left + 1, span.right + 1
        seg_length = end - start

        full_seq[start:end] = seq[1 : 1 + seg_length]
        if zero_out:
            full_seq[1:start] = 0
            full_seq[end:eos_loc] = 0

        if replace_pad:
            pad_length = len(full_seq) - eos_loc
            full_seq[eos_loc:] = seq[1 + seg_length : 1 + seg_length + pad_length]

        # shape: (768, 77) -> (77, 768)
        return full_seq.transpose(0, dim)

    def extend_str(self, nps: List[str]) -> torch.Tensor:
        nps = self._extend_string(nps)

        input_ids = self.tokenize(nps).input_ids
        enc_output = self.text_encoder(input_ids.to(self.device))
        c = enc_output.last_hidden_state
        return c

    def extend_seq(self, nps: List[str]):

        input_ids = self.tokenize(nps).input_ids

        # repeat each NP after embedding
        nps_length = [len(ids) - 2 for ids in input_ids]  # not including bos eos

        enc_output = self.text_encoder(input_ids.to(self.device))
        c = enc_output.last_hidden_state

        # shape: (num_nps, model_max_length, hidden_dim)
        c = torch.stack(
            [c[0]]
            + [self._expand_sequence(seq, l) for seq, l in zip(c[1:], nps_length[1:])]
        )
        return c

    def align_seq(self, nps: List[str], spans: List[Span]) -> KeyValueTensors:

        input_ids = self.tokenize(nps).input_ids
        nps_length = [len(ids) - 2 for ids in input_ids]
        enc_output = self.text_encoder(input_ids.to(self.device))
        c = enc_output.last_hidden_state

        # shape: (num_nps, model_max_length, hidden_dim)
        k_c = [c[0]] + [
            self._align_sequence(c[0].clone(), seq, span, nps_length[0] + 1)
            for seq, span in zip(c[1:], spans[1:])
        ]

        # shape: (num_nps, model_max_length, hidden_dim)
        v_c = [c[0]] + [
            self._align_sequence(c[0].clone(), seq, span, nps_length[0] + 1)
            for seq, span in zip(c[1:], spans[1:])
        ]

        return KeyValueTensors(
            k=[k.unsqueeze(dim=0) for k in k_c],
            v=[v.unsqueeze(dim=0) for v in v_c],
        )

    def apply_text_encoder(
        self,
        struct_attention: STRUCT_ATTENTION_TYPE,
        prompt: str,
        nps: List[str],
        spans: Optional[List[Span]] = None,
    ) -> Union[torch.Tensor, KeyValueTensors]:

        if struct_attention == "extend_str":
            return self.extend_str(nps=nps)

        elif struct_attention == "extend_seq":
            return self.extend_seq(nps=nps)

        elif struct_attention == "align_seq" and spans is not None:
            return self.align_seq(nps=nps, spans=spans)

        elif struct_attention == "none":
            text_input = self.tokenize(prompt)
            return self.text_encoder(text_input.input_ids.to(self.device))[0]

        else:
            raise ValueError(f"Invalid type of struct attention: {struct_attention}")

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        num_images_per_sample: int = 1,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        struct_attention: STRUCT_ATTENTION_TYPE = "none",
        **kwargs,
    ) -> StableDiffusionPipelineOutput:

        if isinstance(prompt, str):
            batch_size = num_images_per_sample
        else:
            raise ValueError(f"`prompt` has to be of type `str` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        preprocessed_prompt = self.preprocess_prompt(prompt)

        doc = self.nlp(preprocessed_prompt)
        tree = Tree.fromstring(str(doc.sentences[0].constituency))

        all_nps = self.get_all_nps(tree=tree, full_sent=preprocessed_prompt)

        cond_embeddings = self.apply_text_encoder(
            struct_attention=struct_attention,
            prompt=preprocessed_prompt,
            nps=all_nps.nps,
            spans=all_nps.spans,
        )

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_input = self.tokenize([""] * batch_size)
            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(self.device)
            )[0]

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            if struct_attention == "align_seq":
                # shape (uncond_embeddings): (batch_size, model_max_length, hidden_dim)
                assert len(uncond_embeddings.size()) == 3
                assert uncond_embeddings.size(dim=0) == batch_size

                # shape (cond_embeddings):
                # KeyValueTensors.k List[(1, num_nps, model_max_length, hidden_dim)]
                # KeyValueTensors.v List[(1, num_nps, model_max_length, hidden_dim)]
                text_embeddings = (uncond_embeddings, cond_embeddings)
            else:
                # shape (uncond_embeddings): (batch_size, model_max_length, hidden_dim)
                # shape (cond_embeddings): (batch_size, num_nps, model_max_length, hidden_dim)
                # shape: (batch_size, 1 + num_nps, model_max_length, hidden_dim)
                breakpoint()
                text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

        # get the initial random noise unless the user supplied it

        # Unlike in other pipelines, latents need to be generated in the target device
        # for 1-to-1 results reproducibility with the CompVis implementation.
        # However this currently doesn't work in `mps`.
        latents_device = "cpu" if self.device.type == "mps" else self.device
        latents_shape = (batch_size, self.unet.in_channels, height // 8, width // 8)
        if latents is None:
            latents = torch.randn(
                latents_shape,
                generator=generator,
                device=latents_device,
            )
        else:
            if latents.shape != latents_shape:
                raise ValueError(
                    f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}"
                )
        latents = latents.to(self.device)

        # set timesteps
        accepts_offset = "offset" in set(
            inspect.signature(self.scheduler.set_timesteps).parameters.keys()
        )
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
        if isinstance(self.scheduler, LMSDiscreteScheduler):
            latents = latents * self.scheduler.sigmas[0]

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                sigma = self.scheduler.sigmas[i]
                # the model input needs to be scaled to match the continuous ODE formulation in K-LMS
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # compute the previous noisy sample x_t -> x_t-1
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                latents = self.scheduler.step(
                    noise_pred, i, latents, **extra_step_kwargs
                ).prev_sample
            else:
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        # run safety checker
        safety_cheker_input = self.feature_extractor(
            self.numpy_to_pil(image), return_tensors="pt"
        ).to(self.device)
        image, has_nsfw_concept = self.safety_checker(
            images=image, clip_input=safety_cheker_input.pixel_values
        )

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept
        )
