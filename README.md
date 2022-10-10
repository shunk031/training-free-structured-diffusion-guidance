# Training-free Structured Diffusion Guidance for Compositional Text-to-Image Synthesis

[![ICLR 2023 OpenReview](http://img.shields.io/badge/ICLR%202023-under%20review-B31B1B.svg)](https://openreview.net/forum?id=PUIqjT4rzq7)
[![CI](https://github.com/shunk031/training-free-structured-diffusion-guidance/actions/workflows/ci.yaml/badge.svg)](https://github.com/shunk031/training-free-structured-diffusion-guidance/actions/workflows/ci.yaml)

Unofficial [🤗 huggingface/diffusers](https://github.com/huggingface/diffusers)-based implementation of the paper *Training-Free Structured Diffusion Guidance for Compositional Text-to-Image Synthesis*. We refer to the author's original implementation as [supplemented in the OpenReview](https://openreview.net/attachment?id=PUIqjT4rzq7&name=supplementary_material).
There is no direct relationship between this implementation and the author.

## TL;DR

> The author proposes a training-free approach to incorporate language structured for compositional text-to-image synthesis

![](./assets/figure1.png)
*Figure 1: Three challenging phenomena in the compositional generation.Attribute leakage:The attribute of one object is (partially) observable in another object. Interchanged attributes: theattributes of two or more objects are interchanged. Missing objects: one or more objects are missing.With slight abuse of attribute binding definitions, we aim to address all three problems in this work.*

## **Training-Free Structured Diffusion Guidance for Compositional Text-to-Image Synthesis**
  - Anonymous authors
  - ICLR 2023 under review

> Large-scale diffusion models have demonstrated remarkable performance on text-to-image synthesis (T2I). Despite their ability to generate high-quality and creative images, users still observe images that do not align well with the text input, especially when involving multiple objects. In this work, we strive to improve the compositional skills of existing large-scale T2I models, specifically more accurate attribute binding and better image compositions. We propose to incorporate language structures with the cross-attention layers based on a recently discovered property of diffusion-based T2I models. Our method is implemented on a state-of-the-art model, Stable Diffusion, and achieves better compositional skills in both qualitative and quantitative results. Our structured cross-attention design is also efficient that requires no additional training samples. Lastly, we conduct an in-depth analysis to reveal potential causes of incorrect image compositions and justify the properties of cross-attention layers in the generation process. 

## Installation

```shell
pip install git+https://github.com/shunk031/training-free-structured-diffusion-guidance
```

## How to use Training-Free Structured Diffusion Guidance (TFSDG)

```python
from tfsdg.pipelines import TFSDGPipeline

pipe = TFSDGPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", use_auth_token=True
)
pipe = pipe.to("cuda")

prompt = "A red car and a white sheep"
image = pipe(prompt, struct_attention="align_seq").images[0]
image.save('a_red_car_and_a_white_sheep.png')
```

## Citation

```bibtex
@misc{kitada-2022-tfsdg,
  author = {Shunsuke Kitada},
  title = {Training-Free Structured Diffusion Guidance for Compositional Text-to-Image Synthesis},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/shunk031/training-free-structured-diffusion-guidance}}
}
```
