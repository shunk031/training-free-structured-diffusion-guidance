import os
from typing import List, Tuple, get_args

import pytest
import torch
import torch.nn as nn
from diffusers.models.attention import CrossAttention
from diffusers.pipelines import StableDiffusionPipeline
from nltk.tree import Tree
from PIL import Image, ImageChops
from tfsdg.models.attention import StructuredCrossAttention
from tfsdg.pipelines import TFSDGPipeline
from tfsdg.pipelines.tfsdg_pipeline import AllNPs, Span, SubNP
from tfsdg.utils.attention_utils import KeyValueTensors
from tfsdg.utils.pipeline_utils import STRUCT_ATTENTION_TYPE


@pytest.fixture
def model_name() -> str:
    return "CompVis/stable-diffusion-v1-4"


@pytest.fixture
def gpu_device() -> str:
    return "cuda"


def _compare_cross_attention(
    proposed_module: nn.Module,
    original_module: nn.Module,
    proposed_name: str,
    original_name: str,
) -> None:

    it = zip(dir(proposed_module), dir(original_module))
    for proposed_attr_str, original_attr_str in it:

        proposed_target_attr = getattr(proposed_module, proposed_attr_str)
        original_target_attr = getattr(original_module, original_attr_str)

        cond1 = isinstance(proposed_target_attr, StructuredCrossAttention)
        cond2 = isinstance(original_target_attr, CrossAttention)

        if cond1:
            if cond2:
                proposed_params = list(proposed_target_attr.parameters())
                original_params = list(original_target_attr.parameters())
                assert len(proposed_params) == len(original_params)

                it_params = zip(proposed_params, original_params)
                for proposed_param, original_param in it_params:
                    if proposed_param.data.ne(original_param.data).sum() > 0:
                        raise ValueError(
                            "Mismatch the weight of the cross attention layer: \n"
                            f"Original: {original_param.data}\n"
                            f"Proposed: {proposed_param.data}"
                        )
            else:
                raise ValueError(
                    "Mismatch cross attention layer: \n"
                    f"Proposed: {proposed_name}({type(proposed_target_attr)}) "
                    f"!= Original: {original_name}({type(original_target_attr)})"
                )

    proposed_tuple: Tuple[str, nn.Module]
    original_tuple: Tuple[str, nn.Module]

    it = zip(proposed_module.named_children(), original_module.named_children())  # type: ignore
    for proposed_tuple, original_tuple in it:  # type: ignore

        proposed_name, proposed_intermediate_child_module = proposed_tuple
        original_name, original_intermediate_child_module = original_tuple

        _compare_cross_attention(
            proposed_module=proposed_intermediate_child_module,
            proposed_name=proposed_name,
            original_module=original_intermediate_child_module,
            original_name=original_name,
        )


def _compare_generated_images(img1: Image.Image, img2: Image.Image) -> None:
    diff = ImageChops.difference(img1, img2)
    assert diff.getbbox() is None, "generated images are different"


def test_cross_attention(model_name: str):

    proposed_pipe = TFSDGPipeline.from_pretrained(model_name)
    original_pipe = StableDiffusionPipeline.from_pretrained(model_name)

    _compare_cross_attention(
        proposed_module=proposed_pipe.unet,
        proposed_name="unet",
        original_module=original_pipe.unet,
        original_name="unet",
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="No GPUs available for testing"
)
def test_txt2img(model_name: str, gpu_device: str):

    #
    # create generator and set seed to the generator
    #
    generator = torch.Generator(device=gpu_device)
    generator = generator.manual_seed(0)

    #
    # create original/proposed pipelines from the pre-trained weights
    #
    pipe_original = StableDiffusionPipeline.from_pretrained(
        model_name,
        use_auth_token=True,
    )
    pipe_proposed = TFSDGPipeline.from_pretrained(
        model_name,
        use_auth_token=True,
    )

    #
    # use the basic prompt for testing
    #
    prompt = "a photo of an astronaut riding a horse on mars"

    #
    # inference using the original pipeline
    #
    pipe_original = pipe_original.to(gpu_device)
    image_original = pipe_original(prompt, generator=generator).images[0]

    #
    # inference using the tfsdg pipeline
    #
    pipe_proposed = pipe_proposed.to(gpu_device)
    image_proposed = pipe_proposed(
        prompt,
        generator=generator,
        struct_attention="none",  # none means the same calculation is done by the TFSDG pipeline
    ).images[0]

    _compare_generated_images(img1=image_original, img2=image_proposed)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="No GPUs available for testing"
)
@pytest.mark.parametrize(
    "in_prompt,",
    (
        "A red car and a white sheep.",
        "A brown bench sits in front of an old white building.",
        "A blue backpack and a brown elephant.",
    ),
)
def test_comparison_txt2img(
    in_prompt: str,
    model_name: str,
    gpu_device: str,
    num_images: int = 10,
):

    pipe_original = StableDiffusionPipeline.from_pretrained(model_name).to(gpu_device)
    pipe_proposed = TFSDGPipeline.from_pretrained(model_name).to(gpu_device)

    preprocessed_prompt = pipe_proposed.preprocess_prompt(in_prompt)

    generator = torch.Generator(device=gpu_device)

    for i in range(num_images):
        seed = generator.seed()
        generator = generator.manual_seed(seed)

        pipe_kwargs = {
            "prompt": in_prompt,
            "generator": generator,
        }
        out_original = pipe_original(**pipe_kwargs)
        out_proposed = pipe_proposed(**pipe_kwargs, struct_attention="align_seq")

        img_original = out_original.images[0]
        img_proposed = out_proposed.images[0]

        sub_dir = "-".join(preprocessed_prompt.split())
        os.makedirs(sub_dir, exist_ok=True)
        img_original_path = os.path.join(sub_dir, f"{i:02d}_original_seed-{seed}.png")
        img_proposed_path = os.path.join(sub_dir, f"{i:02d}_proposed_seed-{seed}.png")

        img_original.save(img_original_path)
        img_proposed.save(img_proposed_path)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="No GPUs available for testing"
)
@pytest.mark.parametrize("struct_attention,", get_args(STRUCT_ATTENTION_TYPE))
def test_pipeline(struct_attention: str, model_name: str, gpu_device: str):

    #
    # create generator and set seed to the generator
    #
    generator = torch.Generator(device=gpu_device)
    generator = generator.manual_seed(42)

    pipe = TFSDGPipeline.from_pretrained(
        model_name,
        use_auth_token=True,
    )
    pipe = pipe.to("cuda")

    #
    # use the prompt from the original tfsdg paper
    #
    prompt = "A red car and a white sheep"

    #
    # test each struct attention type
    #

    output = pipe(prompt, generator=generator, struct_attention=struct_attention)
    image = output.images[0]
    image.save(f"{struct_attention}-a_red_car_and_a_white_sheep.png")


@pytest.mark.parametrize(
    "in_prompt, expected",
    (
        (
            "A red car and a white sheep.",
            "a red car and a white sheep",
        ),
        (
            "A brown bench sits in front of an old white building.",
            "a brown bench sits in front of an old white building",
        ),
        (
            "A blue backpack and a brown elephant.",
            "a blue backpack and a brown elephant",
        ),
    ),
)
def test_preprocess_prompt(in_prompt: str, expected: str):

    pipe = TFSDGPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        use_auth_token=True,
    )

    assert pipe.preprocess_prompt(in_prompt) == expected


@pytest.mark.parametrize(
    "in_prompt, expected_nps",
    (
        (
            "A red car and a white sheep.",
            [
                SubNP(text="a red car and a white sheep", span=Span(left=0, right=7)),
                SubNP(text="a red car", span=Span(left=0, right=3)),
                SubNP(text="a white sheep", span=Span(left=4, right=7)),
            ],
        ),
        (
            "A brown bench sits in front of an old white building.",
            [
                SubNP(text="a brown bench", span=Span(left=0, right=3)),
                SubNP(
                    text="front of an old white building", span=Span(left=5, right=11)
                ),
                SubNP(text="an old white building", span=Span(left=7, right=11)),
            ],
        ),
        (
            "A blue backpack and a brown elephant.",
            [
                SubNP(
                    text="a blue backpack and a brown elephant",
                    span=Span(left=0, right=7),
                ),
                SubNP(text="a blue backpack", span=Span(left=0, right=3)),
                SubNP(text="a brown elephant", span=Span(left=4, right=7)),
            ],
        ),
    ),
)
def test_get_sub_nps(in_prompt: str, expected_nps: List[SubNP]):

    pipe = TFSDGPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        use_auth_token=True,
    )

    preprocessed_prompt = pipe.preprocess_prompt(in_prompt)
    doc = pipe.nlp(preprocessed_prompt)
    tree = Tree.fromstring(str(doc.sentences[0].constituency))

    all_sub_nps = pipe.get_sub_nps(tree=tree, left=0, right=len(tree.leaves()))
    assert len(all_sub_nps) == len(expected_nps)

    for sub_nps, expected_np in zip(all_sub_nps, expected_nps):
        assert sub_nps == expected_np, f"{sub_nps} != {expected_np}"


@pytest.mark.parametrize(
    "in_prompt, expected_nps",
    (
        (
            "A red car and a white sheep.",
            AllNPs(
                nps=["a red car and a white sheep", "a red car", "a white sheep"],
                spans=[
                    Span(left=0, right=7),
                    Span(left=0, right=3),
                    Span(left=4, right=7),
                ],
                lowest_nps=[],
            ),
        ),
        (
            "A brown bench sits in front of an old white building.",
            AllNPs(
                nps=[
                    "a brown bench sits in front of an old white building",
                    "a brown bench",
                    "front of an old white building",
                    "an old white building",
                ],
                spans=[
                    Span(left=0, right=11),
                    Span(left=0, right=3),
                    Span(left=5, right=11),
                    Span(left=7, right=11),
                ],
                lowest_nps=[],
            ),
        ),
        (
            "A blue backpack and a brown elephant.",
            AllNPs(
                nps=[
                    "a blue backpack and a brown elephant",
                    "a blue backpack",
                    "a brown elephant",
                ],
                spans=[
                    Span(left=0, right=7),
                    Span(left=0, right=3),
                    Span(left=4, right=7),
                ],
                lowest_nps=[],
            ),
        ),
    ),
)
def test_get_all_nps(in_prompt: str, expected_nps: AllNPs):

    pipe = TFSDGPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        use_auth_token=True,
    )

    preprocessed_prompt = pipe.preprocess_prompt(in_prompt)
    doc = pipe.nlp(preprocessed_prompt)
    tree = Tree.fromstring(str(doc.sentences[0].constituency))

    all_nps = pipe.get_all_nps(tree=tree, full_sent=preprocessed_prompt)
    assert all_nps == expected_nps


@pytest.mark.parametrize(
    "in_prompt, expected_strings",
    (
        (
            "A red car and a white sheep.",
            [
                "a red car and a white sheep",
                " a red car a red car a red car a red car a red car a red car a red car a red car a red car a red car a red car a red car a red car a red car a red car a red car a red car a red car a red car a red car a red car a red car a red car a red car a red car",
                " a white sheep a white sheep a white sheep a white sheep a white sheep a white sheep a white sheep a white sheep a white sheep a white sheep a white sheep a white sheep a white sheep a white sheep a white sheep a white sheep a white sheep a white sheep a white sheep a white sheep a white sheep a white sheep a white sheep a white sheep a white sheep",
            ],
        ),
        (
            "A brown bench sits in front of an old white building.",
            [
                "a brown bench sits in front of an old white building",
                " a brown bench a brown bench a brown bench a brown bench a brown bench a brown bench a brown bench a brown bench a brown bench a brown bench a brown bench a brown bench a brown bench a brown bench a brown bench a brown bench a brown bench a brown bench a brown bench a brown bench a brown bench a brown bench a brown bench a brown bench a brown bench",
                " front of an old white building front of an old white building front of an old white building front of an old white building front of an old white building front of an old white building front of an old white building front of an old white building front of an old white building front of an old white building front of an old white building front of an old white building",
                " an old white building an old white building an old white building an old white building an old white building an old white building an old white building an old white building an old white building an old white building an old white building an old white building an old white building an old white building an old white building an old white building an old white building an old white building an old white building",
            ],
        ),
        (
            "A blue backpack and a brown elephant.",
            [
                "a blue backpack and a brown elephant",
                " a blue backpack a blue backpack a blue backpack a blue backpack a blue backpack a blue backpack a blue backpack a blue backpack a blue backpack a blue backpack a blue backpack a blue backpack a blue backpack a blue backpack a blue backpack a blue backpack a blue backpack a blue backpack a blue backpack a blue backpack a blue backpack a blue backpack a blue backpack a blue backpack a blue backpack",
                " a brown elephant a brown elephant a brown elephant a brown elephant a brown elephant a brown elephant a brown elephant a brown elephant a brown elephant a brown elephant a brown elephant a brown elephant a brown elephant a brown elephant a brown elephant a brown elephant a brown elephant a brown elephant a brown elephant a brown elephant a brown elephant a brown elephant a brown elephant a brown elephant a brown elephant",
            ],
        ),
    ),
)
def test_extend_string(in_prompt: str, expected_strings: List[str]):

    pipe = TFSDGPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        use_auth_token=True,
    )

    preprocessed_prompt = pipe.preprocess_prompt(in_prompt)
    doc = pipe.nlp(preprocessed_prompt)
    tree = Tree.fromstring(str(doc.sentences[0].constituency))

    all_nps = pipe.get_all_nps(tree=tree, full_sent=preprocessed_prompt)

    extended_strings = pipe._extend_string(nps=all_nps.nps)

    assert len(extended_strings) == len(expected_strings)
    for extended_string, expected_string in zip(extended_strings, expected_strings):
        assert (
            extended_string == expected_string
        ), f"{extended_string} != {expected_string}"


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="No GPUs available for testing"
)
@pytest.mark.parametrize(
    "in_prompt,",
    (
        "A red car and a white sheep.",
        "A brown bench sits in front of an old white building.",
        "A blue backpack and a brown elephant.",
    ),
)
def test_expand_sequence(in_prompt: str, gpu_device: str):

    pipe = TFSDGPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        use_auth_token=True,
    )
    pipe = pipe.to(gpu_device)

    preprocessed_prompt = pipe.preprocess_prompt(in_prompt)
    doc = pipe.nlp(preprocessed_prompt)
    tree = Tree.fromstring(str(doc.sentences[0].constituency))

    all_nps = pipe.get_all_nps(tree=tree, full_sent=preprocessed_prompt)
    nps = all_nps.nps

    input_ids = pipe.tokenize(nps).input_ids
    nps_length = [len(ids) - 2 for ids in input_ids]

    enc_output = pipe.text_encoder(input_ids.to(pipe.device))
    c = enc_output.last_hidden_state

    assert c[1:].size(0) == len(nps_length)

    for seq, np_length in zip(c[1:], nps_length[1:]):
        assert tuple(seq.size()) == (77, 768), seq.size()
        assert np_length == 77 - 2, np_length  # not including BOS and EOS

        with torch.no_grad():
            expanded_seq = pipe._expand_sequence(seq, np_length)

        assert tuple(expanded_seq.size()) == (77, 768), expanded_seq.size()


@pytest.mark.parametrize(
    "in_prompt,",
    (
        "A red car and a white sheep.",
        "A brown bench sits in front of an old white building.",
        "A blue backpack and a brown elephant.",
    ),
)
def test_align_sequence(in_prompt: str):

    pipe = TFSDGPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        use_auth_token=True,
    )

    preprocessed_prompt = pipe.preprocess_prompt(in_prompt)
    doc = pipe.nlp(preprocessed_prompt)
    tree = Tree.fromstring(str(doc.sentences[0].constituency))

    all_nps = pipe.get_all_nps(tree=tree, full_sent=preprocessed_prompt)
    nps = all_nps.nps
    spans = all_nps.spans

    input_ids = pipe.tokenize(nps).input_ids
    nps_length = [len(ids) - 2 for ids in input_ids]

    enc_output = pipe.text_encoder(input_ids.to(pipe.device))
    c = enc_output.last_hidden_state

    assert c[1:].size(0) == len(spans[1:])

    for seq, span in zip(c[1:], spans[1:]):
        aligned_seq = pipe._align_sequence(
            full_seq=c[0].clone(), seq=seq, span=span, eos_loc=nps_length[0] + 1
        )
        assert tuple(aligned_seq.size()) == (77, 768), aligned_seq.size()


@pytest.mark.parametrize(
    "in_prompt,",
    (
        "A red car and a white sheep.",
        "A brown bench sits in front of an old white building.",
        "A blue backpack and a brown elephant.",
    ),
)
def test_extend_str(in_prompt: str):
    pipe = TFSDGPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        use_auth_token=True,
    )

    preprocessed_prompt = pipe.preprocess_prompt(in_prompt)
    doc = pipe.nlp(preprocessed_prompt)
    tree = Tree.fromstring(str(doc.sentences[0].constituency))

    all_nps = pipe.get_all_nps(tree=tree, full_sent=preprocessed_prompt)
    c = pipe.extend_str(nps=all_nps.nps)

    assert c.size(0) == len(all_nps.nps)
    assert c.size(1) == 77 and c.size(2) == 768


@pytest.mark.parametrize(
    "in_prompt,",
    (
        "A red car and a white sheep.",
        "A brown bench sits in front of an old white building.",
        "A blue backpack and a brown elephant.",
    ),
)
def test_extend_seq(in_prompt: str):
    pipe = TFSDGPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        use_auth_token=True,
    )

    preprocessed_prompt = pipe.preprocess_prompt(in_prompt)
    doc = pipe.nlp(preprocessed_prompt)
    tree = Tree.fromstring(str(doc.sentences[0].constituency))

    all_nps = pipe.get_all_nps(tree=tree, full_sent=preprocessed_prompt)

    with torch.no_grad():
        c = pipe.extend_seq(nps=all_nps.nps)

    assert c.size(0) == len(all_nps.nps)
    assert c.size(1) == 77 and c.size(2) == 768


@pytest.mark.parametrize(
    "in_prompt,",
    (
        "A red car and a white sheep.",
        "A brown bench sits in front of an old white building.",
        "A blue backpack and a brown elephant.",
    ),
)
def test_align_seq(in_prompt: str):
    pipe = TFSDGPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        use_auth_token=True,
    )

    preprocessed_prompt = pipe.preprocess_prompt(in_prompt)
    doc = pipe.nlp(preprocessed_prompt)
    tree = Tree.fromstring(str(doc.sentences[0].constituency))

    all_nps = pipe.get_all_nps(tree=tree, full_sent=preprocessed_prompt)
    c = pipe.align_seq(nps=all_nps.nps, spans=all_nps.spans)

    assert isinstance(c, KeyValueTensors), c

    assert len(all_nps.nps) == c.k.size(0), c.k.size()
    assert c.k.size(1) == pipe.tokenizer.model_max_length, c.k.size()
    assert c.k.size(2) == pipe.text_encoder.config.hidden_size, c.k.size()

    assert len(all_nps.nps) == c.v.size(0), c.v.size()
    assert c.v.size(1) == pipe.tokenizer.model_max_length, c.v.size()
    assert c.v.size(2) == pipe.text_encoder.config.hidden_size, c.v.size()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="No GPUs available for testing"
)
@pytest.mark.parametrize("struct_attention,", get_args(STRUCT_ATTENTION_TYPE))
def test_apply_text_encoder(
    struct_attention: str,
    model_name: str,
    gpu_device: str,
    in_prompt: str = "A red car and a white sheep.",
):
    pipe = TFSDGPipeline.from_pretrained(model_name, use_auth_token=True)
    pipe = pipe.to(gpu_device)

    preprocessed_prompt = pipe.preprocess_prompt(in_prompt)
    doc = pipe.nlp(preprocessed_prompt)
    tree = Tree.fromstring(str(doc.sentences[0].constituency))

    all_nps = pipe.get_all_nps(tree=tree, full_sent=preprocessed_prompt)

    with torch.no_grad():
        c = pipe.apply_text_encoder(
            struct_attention=struct_attention,
            prompt=preprocessed_prompt,
            nps=all_nps.nps,
            spans=all_nps.spans,
        )

    if struct_attention == "align_seq":
        assert isinstance(c, KeyValueTensors)

        assert c.k.size(0) == len(all_nps.nps), c.k.size()
        assert c.k.size(1) == pipe.tokenizer.model_max_length, c.k.size()
        assert c.k.size(2) == pipe.text_encoder.config.hidden_size, c.k.size()

        assert c.v.size(0) == len(all_nps.nps), c.v.size()
        assert c.v.size(1) == pipe.tokenizer.model_max_length, c.v.size()
        assert c.v.size(2) == pipe.text_encoder.config.hidden_size, c.v.size()
    else:
        assert isinstance(c, torch.Tensor)
        assert c.size(0) == len(all_nps.nps), c.size()
        assert c.size(1) == pipe.tokenizer.model_max_length, c.size()
        assert c.size(2) == pipe.text_encoder.config.hidden_size, c.size()
