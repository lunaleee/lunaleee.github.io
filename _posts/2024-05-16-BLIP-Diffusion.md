---
title: "[논문 리뷰] BLIP-Diffusion: Pre-trained Subject Representation for Controllable Text-to-Image Generation and Editing"
author: lunalee
date: 2024-05-16 19:41:11 +0900
categories: [AI, Paper Review]
tags: [Multi-modal, VLP]
pin: false
math: true
---

<br/><br/>
`Salesforce AI Research`  `arXiv 2023`

- Paper: [https://arxiv.org/abs/2305.14720](https://arxiv.org/abs/2305.14720)
- Git: [https://github.com/salesforce/LAVIS/tree/main/projects/blip-diffusion](https://github.com/salesforce/LAVIS/tree/main/projects/blip-diffusion)
- Page: [https://dxli94.github.io/BLIP-Diffusion-website/](https://dxli94.github.io/BLIP-Diffusion-website/)
<br/><br/><br/>

#### 📖 핵심 훑어보기 !!

- Subject-driven text-to-image generation을 목적으로, 일반적인 subject representation을 학습하는 것을 목표로 하는 BLIP-Diffusion 모델 제안
- 일반적인 subject representation을 학습하기 위한 two-stage pre-training 전략 제시
    1.  multimodal representation learning: BLIP-2를 적용, 입력 이미지를 기반으로 text-aligned visual feature를 생성
    2. subject representation learning: 1-stage에서 생성한 subject representation을 이용해 diffusion 모델이 새로운 변형 이미지를 생성하는 방법을 학습
- Pre-train된 BLIP-Diffusion 모델을 foundation generation 모델로 사용하고, 다양한 기존 모델을 통합하여 추가적인 학습 없이 다양한 task에 적용(ControlNet, Prompt-to-prompt)
<br/><br/><br/><br/>

# Introduction

---

해당 논문은 상당 부분 BLIP-2에 기반하므로 BLIP-2에 대한 사전 지식이 필요하신 분은 블로그 내의 [[BLIP-2 논문 리뷰](https://lunaleee.github.io/posts/BLIP-2/)] 글을 참고해주세요!
<br/><br/>

![BLIP-Diffusion_1.png](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/b339e350-415b-4c50-bc8e-8b636fd165f2){: width="900px"}

**<mark style='background-color: var(--hl-yellow)'><span style='color: var(--text-color)'>Subject-driven(주제 기반) generation</span></mark>**은 Text-to-image generation task 중 하나로 **입력 subject(주제)의 모양을 유지**하면서 **이미지를 새롭게 변형하는 것**을 의미한다(위 그림 참조). 일반적으로 Pre-train된 text-image genration 모델을 사용하여, 특정 text embedding을 변경해가며 이에 해당하는 image 집합을 재구성하는 방식으로 학습을 진행한다. 이와 같은 방법은 각각의 subject에 대해 fine-tuning step을 거쳐하기 때문에, 광범위한 subject에 대해 확장하기 어려운 문제가 있다. 
<br/><br/>

저자는 이러한 문제가 대부분의 pre-train된 text-to-image 모델이 image와 text 모두를 control 입력으로 사용할 수 있는 **multimodal control이 불가능**하기 때문이라고 한다. Subject visual을 높은 충실도로 캡처하면서, text space와 잘 align되는 subject representation을 학습하는 것이 어렵다는 것이다. 이러한 문제를 극복하기 위해 본 논문에서는 **<mark style='background-color: var(--hl-yellow)'><span style='color: var(--text-color)'>subject-driven text-to-image generation을 위한 BLIP-Diffusion</span></mark>**을 제안한다. BLIP-Diffusion은 zero-shot 또는 few-step fine-tuning만으로도 subject-driven generation을 가능하게 하는 **<mark style='background-color: var(--hl-green)'><span style='color: var(--text-color)'>generic subject representation</span></mark>**을 학습하는 것을 목표로 한다*.* 
<br/><br/><br/>

BLIP-Diffusion에서는 generic subject representation 학습을 위한 two-stage pre-training 전략을 제시했다. 

1. BLIP-2를 적용, 입력 이미지를 기반으로 text-aligned visual feature를 생성하는 **<mark style='background-color: var(--hl-green)'><span style='color: var(--text-color)'>multimodal representation learning</span></mark>** 수행
2. Diffusion 모델이 subject 기반 새로운 변형 이미지를 생성하는 방법을 학습하는 **<mark style='background-color: var(--hl-green)'><span style='color: var(--text-color)'>subject representation learning task</span></mark>** 설계
<br/><br/><br/><br/><br/><br/>

# Method

---

BLIP-Diffusion은 pre-train된 subject representation을 통해 multimodal control을 가능하게 하는 모델이다. 이를 위해 subject-specific visual appearance를 포착하는 동시에 text prompt와 일치하는 subject representation을 학습하는 것을 목표로 한다. 위에서 언급한 대로 two-stage pre-training 전략을 사용했다.
<br/><br/>

## 1. Multimodal Representation Learning with BLIP-2

![BLIP-Diffusion_2.png](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/b4c2200c-6c20-495d-a75f-68c128634bfa){: width="400px"}

먼저 생성모델로 **Stable Diffusion** 모델(참조: [Stable Diffusion 논문 리뷰](https://lunaleee.github.io/posts/StableDiffusion/))을 사용한다. 위 그림처럼 text embedding은 CLIP에서 생성되어 전달된다. 이 때 prompt로 사용되는 subject representation과 text가 서로 잘 align 되어있는 것이 중요하다. 

논문에서는 vision-language pre-trained 모델인 **<mark style='background-color: var(--hl-yellow)'><span style='color: var(--text-color)'>BLIP-2</span></mark>**를 사용하여 high-quality **<mark style='background-color: var(--hl-yellow)'><span style='color: var(--text-color)'>text-aligned visual representation</span></mark>**를 생성한다.
<br/><br/><br/>

![BLIP-Diffusion_3.png](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/63899b32-eb88-44c4-9008-3f9b6fbb0799){: width="500px"}

구체적으로는 위 그림과 같이, **frozen pre-trained Image Encoder**와 **Q-Former**(multi-modal encoder, BLIP-2 논문리뷰 참조)로 구성되어있다. 

BLIP-2와 마찬가지로 Q-Former는 고정된 수의 learnable query token을 사용한다. Query 집합은 self-attention layer를 통해 text와 상호작용하고, cross-attention layer를 통해 image feature와 상호작용하여 출력으로 text-aligned image feature를 생성한다(BLIP-2와 동일).

저자는 기존 BLIP-2와 같이 query를 32개로 구현하면(=output feature도 32개), output feature가 CLIP text embedding에 비해 너무 강해져 이미지 생성에 적절하게 조합되지 않으므로, query token의 수를 절반인 16개로 구현했다고 한다.
<br/><br/>

또한 BLIP-2 pre-training 방법과 동일하게 ITC, ITG, ITM loss를 모두 사용하여 학습을 진행했다. 결과적으로, 일반적인 image-text paired data에 대한 multimodal representation learning을 통해 모델이 다양한 visual, textual concept을 학습할 수 있도록 한다.
<br/><br/><br/><br/><br/><br/>

## 2. Subject Representation Learning with Stable Diffusion

Multimodal representation learning을 통해 입력 이미지에 대한 일반적인 의미 정보를 추출하였으므로, 다음으로는 Diffusion 모델이 이러한 visual representation을 활용하여 subject의 변형 이미지를 생성할 수 있도록 학습하는 것을 목표로 한다.
<br/><br/><br/>

#### Model Architecture.

![BLIP-Diffusion_4.png](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/e7c87f11-89ab-44e0-9e16-cec0bd98b657){: width="700px"}

전체적인 모델 구조는 위의 그림과 같다. Subject Representation Learning은 아래와 같은 단계를 거쳐 학습된다. 
<br/><br/>

- (Multimodal representation learning stage) BLIP-2의 Q-Former(multi-modal encoder)는 pre-train 과정 중 subject 이미지, subject category가 포함된 text를 입력으로 받아 category-aware subject visual representation을 생성했다.
- Q-Former의 출력, 즉 subject representation은 feed-forward layer(두 개의 Linear layer와 GELU로 구성)를 사용하여 변환된다.
- 변환된 feature는 text prompt token embedding에 **<mark style='background-color: var(--hl-green)'><span style='color: var(--text-color)'>soft visual subject prompt</span></mark>**로 추가된다. 이 때 아래와 같은 template을 사용한다.
<br/>

$$
\text{“[text prompt], the [subject text] is [subject prompt]"}
$$

- 마지막으로, 결합된 text / subject embedding은 CLIP text encoder로 전달되어 diffusion 모델이 이미지를 생성하는 guidance 역할을 한다.

<blockquote class="prompt-info"> 여기서 soft visual prompt는 기본 diffusion 모델에 최소한의 구조적 변경을 가하여 subject representation을 주입하는 동시에, 기본 diffusion 모델의modeling capability를 물려받을 수 있는 방법이라고 한다.</blockquote>
<br/><br/><br/>

#### Subject-generic Pre-training with Prompted Context Generation.

입력 이미지에서 generic한 subject를 학습하기 위해 기존의 방법들은 multi-modal encoder(본 논문에서는 Q-Former)의 입력과 diffusion model의 입력으로 동일한 이미지를 사용했다. 저자가 진행한 사전 실험에서 이와 같은 방법은 입력의 배경에 의해 크게 간섭받거나, 입력과 별반 다를바 없는 이미지를 생성해내는 trivial solution으로 이어지는 문제가 있었다고 한다. 
<br/><br/>

이러한 문제를 해결하기 위해 BLIP-DIffusion에서는 새로운 **<mark style='background-color: var(--hl-yellow)'><span style='color: var(--text-color)'>prompted context generation</span></mark>**을 제안했다. Random한 배경에 subject 이미지를 합성하여 input-target training pair를 생성하고, 모델은 이 합성된 subject 이미지를 입력으로 받은 뒤 text prompt에 따라 원본 subject 이미지를 생성하는 것을 목표로 한다. 구체적으로 아래와 같은 단계를 거친다.
<br/><br/>

- Subject가 포함된 이미지와 해당 category text를 **text-prompted segmentation 모델**인 [CLIPSeg📄](https://arxiv.org/abs/2112.10003)에 넣는다.
- CLIPSeg 출력 segmentation map에서 더 높은 confidence를 가진 부분을 known foreground, 낮은 confidence를 uncertain region, 나머지를 known background로 설정하여 **trimap**을 생성한다.
- Trimap이 주어지면, closed-form matting을 사용하여 foreground, 즉 subject를 추출한다.
- 추출된 subject를 alpha blending을 사용하여 random한 background 이미지에 합성한다.
- 합성 이미지를 입력으로 사용하고 원래 subject 이미지를 출력으로 사용하여 하나의 학습 이미지 pair로 사용한다.
<br/><br/><br/>

![BLIP-Diffusion_5.png](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/d4f9f8b5-c79f-4dfc-a53e-8164453606d8){: width="600px"}

위의 그림과 같이 합성 pair는 subject와 background를 분리하여 subject와 무관한 정보가 prompt에 인코딩되는 것을 방지한다. 위 방식으로 diffusion 모델은 subject prompt, text prompt를 함께 고려한 pre-train 모델로 학습된다.

Pre-train 중에 image encoder를 frozen하고 diffusion 모델의 text encoder(CLIP)와 U-Net, BLIP-2의 Q-Former를 공동으로 학습한다. 원래의 text-to-image generation 기능을 더 잘 보존하기 위해, diffusion guide로 text prompt만 사용하면서 subject prompt를 15% 확률로 random 삭제했을 때 더 좋은 성능을 보이는 것을 발견했다고 한다.
<br/><br/><br/><br/><br/><br/>

## 3. Fine-tuning and Controllable Inference

이렇게 pre-train된 subject representation은 zero-shot generation 뿐 아니라 특정 custom subject에 대한 fine-tuning도 가능하게 한다. 뿐만 아니라 기존 diffusion 모델의 기능을 활용할 수 있으므로, BLIP-Diffusion 모델을 foundation generation 모델로 사용하고 추가적으로 image generation / editing 기술을 활용할 수도 있다. 
<br/><br/>

#### Subject-specific Fine-tuning and Inference.

먼저  pre-trained generic subject representation을 사용하여 개별적인 subject에 대한 fine-tuning 과정에 대해 알아보자. 아래와 같은 방법으로, 단일 A100 GPU에서 20~40초 정도의 fine-tuning 시간이 걸린다고 한다.

- 몇 개의 subject 이미지와 subject category text가 주어지면, multi-modal encoder(Q-Former)를 사용하여  개별적인 subject representation을 얻는다.
- 그 다음 모든 subject 이미지의 subject representation의 평균으로 subject prompt embedding을 초기화한다. 이 방식으로 fine-tuning 과정 중에는 multi-modal encoder의 학습은 필요하지 않다.
- Diffusion 모델은 text prompt embedding과 평균 subject embedding을 사용하여 target 이미지를 생성하도록 fine-tuning 된다.
- 여기서 diffusion 모델의 text encoder(CLIP)은 frozen 하여 subject 이미지에 대한 overfitting을 막는다.
<br/><br/><br/><br/>

#### Structure-controlled Generation with ControlNet.

![BLIP-Diffusion_6.png](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/38bba073-997c-42d1-95c7-00cfb1c170f1){: width="500px"}

추가적으로 subject-control을 위한 multimodal conditioning mechanism을 도입했다. 이를 위해서 **<mark style='background-color: var(--hl-yellow)'><span style='color: var(--text-color)'>ControlNet</span></mark>**을 통합하여 structure-controlled과 subject-controlled generation을 동시에 가능하게 했다.

위 그림과 같이 pre-train 된 ControlNet의 U-Net 구조를 BLIP-Diffusion에 연결한다. ControlNet 연결을 통해 모델은 subject에 관한 단서 뿐 아니라 input의 structure condition(e.g. edge map, depth map) 또한 고려할 수 있다. 이렇게 다양한 기존 모델을 통합하여 추가적인 학습 없이 이미지 생성이 가능한 것이 BLIP-diffusion의 장점이라고 한다.
<br/><br/>

> [**ControlNet**📄](https://arxiv.org/abs/2302.05543)<br/>
> Stable Diffusion에서 좀 더 세밀한 제어를 위해 제안된 모델. 이미지 생성 과정에서 공간적인 context(e.g. edge maps, segmentation maps, depth 등)를 condition으로 주어 더 세부적인 제어가 가능하다.<br/>
> 구조는 대략적으로 아래 그림과 같이 두 개의 모델로 구성되어 있다. 하나는 기존에 pre-train된 생성 모델(stable diffusion)이고, 하나는 이를 제어하기 위한 조건부 모델이다. 조건부 모델은 사용자가 입력한 condition에 따라 생성 모델의 결과를 조정하는데 사용된다. 자세한 모델의 구조 및 원리는 논문을 참조해보자.
>
> ![BLIP-Diffusion_7.png](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/04446a79-65f5-4eb8-b047-d28e54d6ede9){: width="450px"}
> 

<br/><br/><br/>

#### Subject-driven Editing with Attention Control.

![BLIP-Diffusion_8.png](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/58d3fcc7-0013-4c0e-9418-bc77af9d4b98){: width="650px"}
_그림에서 Text Encoder는 Diffusion 모델의 CLIP, U-Net은 Diffusion 모델(forward/backward process)._

BLIP-Diffusion에서는 multimodal controlled generation을 위해 subject prompt embedding과 text prompt embedding을 결합한다. 여기서 prompt token의 cross-attention map을 조작하여 **subject-driven image editing**을 수행하게 했다. **<mark style='background-color: var(--hl-yellow)'><span style='color: var(--text-color)'>prompt-to-prompt</span></mark>** 논문에서 영감을 받은 방식으로, **cross-attention control technique**을 사용했다.
<br/><br/>

위의 그림과 같이 subject-specific 이미지로 원본 이미지를 편집한다. 편집 과정은 아래의 단계를 거쳐 진행된다.

- 먼저 편집할 text token(e.g. dog)을 지정한다.
- 그 다음 지정된 token의 cross-attention map을 사용하여 편집할 영역에 대한 mask를 자동으로 추출한다.
- 편집되지 않는 영역은 보존하기 위해, subject embedding에 대한 attention map을 생성하는 동안 원본 생성 attention map은 유지한다.
- 추출된 editing mask를 기반으로 각 step에서 denoising latent를 혼합한다. 즉, 편집되지 않은 영역의 latent는 원본 생성에서 나온 것이고, 편집된 영역의 latent는 subject-driven 생성에서 나온 것이다.
<br/><br/>

> [**Prompt-to-prompt**📄](https://arxiv.org/abs/2208.01626)
> 

<br/><br/><br/><br/><br/>

# Experiments

---

## 1. Pre-training Datasets and Details

- Multimodal representation learning
    - BLIP-2 구조를 사용.
    - 129M image-text pair에 대해 pre-train 함. 여기에는 CapFilt caption이 있는  LAION, COCO, Visual Genome, Conceptual Captions의 115M image-text pair가 포함됨.
    - CLIP에서의 image encoder 사용, BERTbase로 Q-Former 초기화.
- Subject representation learning
    - OpenImage-V6 데이터셋에서 특정 subject를 포함하는 292K의 subset을 사용(인간과 관련있는 subject 제외).
    - BLIP-2 OPT를 사용하여 caption을 text prompt로 생성
    - web에서 59K의 background 이미지를 얻어 subject와 합성함
    - Stable Diffusion v1-5를 기본 diffusion 모델로 사용

<br/><br/><br/>

## 2. Experimental Results

#### Main Qualitative Results.

아래 그림에서 BLIP-Diffusion의 정성적 결과를 볼 수 있다.

- row #1: pre-train된 subject representation을 이용한 zero-shot subject-driven generation
- row #3-6: 효율적인 fine-tuning이 가능하므로 다양한 task에 대해서도 높은 충실도로 이미지를 생성
- row #7-8: ControlNet과 결합하여 구조와 subject를 동시에 제어
- row #9-10: subject 정보를 이미지 편집 파이프라인에 도입하여 특정 subject 이미지로 원본 이미지를 편집할 수 있음<br/>
![BLIP-Diffusion_9.png](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/5103f512-124e-4149-aa22-a5765f30e231){: width="700px"}
<br/><br/><br/><br/>

#### Comparisons on DreamBooth Dataset.

BLIP-Diffusion과 다양한 SOTA 생성모델을 DreamBooth 데이터셋에 대해 비교했다. 데이터셋에는 30개의 subject에 대해 각각 4~7개의 이미지가 포함된다. 결과는 아래 그림과 같다.

![BLIP-Diffusion_10.png](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/992cbddc-f124-4664-b84b-821ad99c9c29){: width="900px"}
<br/><br/><br/>

다음으로는 이에 대한 정량적인 결과이다. DINO 및 CLIP-I 점수는 subject alignment을 측정하고 CLIP-T는 image-text alignment을 측정하는 지표이다. 저자는  각 text prompt에 대해 4개의 이미지를 생성하여 모든 subject에 대해 총 3,000개의 이미지를 생성했다. 전반적인 결과는 정성적 결과와 유사하게 BLIP-Diffusion이 우수한 성능을 보이는 것을 알 수 있다. 

![BLIP-Diffusion_11.png](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/77658e20-07a8-4be6-8462-72b71365facd){: width="900px"}
<br/><br/><br/><br/>

#### Ablation Studies.

![BLIP-Diffusion_12.png](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/131cd930-a665-4e19-a0ac-43fa906edb9d){: width="900px"}

다음으로는 250K subject representation learning step을 사용하여 ablation study를 수행했다. 위 표는 zero-shot evaluation 결과를 나타낸다. 결과를 통해 저자는 아래와 같은 결론을 얻었다고 한다.

1. subject embedding과 text prompt embedding 간의 representation gap을 해소하기 위해 multimodal representation learning이 중요하다.
2. Diffusion 모델의 text encoder를 freezing면 subject embedding과 text prompt embedding 간의 상호 작용이 악화되어 text prompt가 무시되는 문제가 발생한다.
3. subject text를 multimodal encoder에 제공하면 클래스별 시각적 사전 정보를 주입하는 데 도움이 되어 성능이 향상된다.
4.  random subject embedding 삭제를 통한 pre-training은 diffusion 모델의 generation ability을 더 잘 보존하는 데 도움이 되어 결과에 도움을 준다.
<br/><br/><br/><br/>

#### Subject Representation Visualization.

![BLIP-Diffusion_13.png](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/88bed676-4ed9-47f4-82e3-f5ed61f2ea8c){: width="1100px"}

이미지 내의 픽셀이 해당되는 emdedding에 반응한다는 관찰 결과에 따라 cross-attention map을 관찰하여 학습된 subject embedding을 시각화하였다. 위의 결과를 보면 학습된 embedding들이 세밀하면서도 각각 다른 측면을 인코딩하는 것을 알 수 있다(e.g. 일부 embedding은 보다 지역적인 feature에, 다른 embeddding은 전체적인 feature를 인코딩). 결론적으로 여러개의 subject embedding을 사용하는 상호 보완적인 효과를 볼 수 있다.
<br/><br/><br/><br/>

#### Zero-shot Subject-driven Image Manipulation.

BLIP-Diffusion은 이미지 생성을 가이드하는 subject feature를 추출할 수 있다. Subject-driven generations / editing 외에도 pre-train된 subject representation을 이용하여 subject-driven style transfer 또는 subject interpolation 등의 task를 수행할 수있다.
<br/><br/>

- Subject-driven Style Transfer<br/>
![BLIP-Diffusion_14.png](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/d5a924cb-08ec-490a-98e4-97a505ac2a9d){: width="900px"}
<br/><br/>

- Subject Interpolation<br/>
![BLIP-Diffusion_15.png](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/4e5c4de4-d313-43f1-b580-ee93fbb2a538){: width="700px"}

<br/><br/><br/><br/>
