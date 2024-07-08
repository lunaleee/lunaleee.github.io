---
title: "[논문 리뷰] Stable Diffusion, High-Resolution Image Synthesis with Latent Diffusion Models"
author: lunalee
date: 2024-02-04 19:32:29 +0700
categories: [AI, Paper Review]
tags: [Image, Generation, Diffusion]
pin: false
math: true
---

<br/>
Stable Diffusion이라 불리는 이 논문은 2022년 발표된 Image Generation 모델이다. 이 모델은 Latent Diffusion 모델의 구조를 갖고 있다. 기존의 Diffusion 기반 모델들과 달리 Stable Diffusion은 고해상도 이미지 합성이 가능하다. 이 글에서 Latent Diffuion의 기본 구조부터 Stable Diffusion 까지에 대한 내용을 정리해보자.
<br/><br/>

`Runway ML` `CVPR 2022`

- Paper: [https://arxiv.org/abs/2112.10752](https://arxiv.org/abs/2112.10752)
- Git: [https://github.com/CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion)
- Project Page: [https://stability.ai/news/stable-diffusion-3](https://stability.ai/news/stable-diffusion-3)
<br/><br/><br/><br/>

# Introduction

---

< 저자의 논문의 흐름을 따라가기 위해 introduction 부분을 풀어서 정리했다. 필요 없다면 이부분은 가볍게 넘어가도록 하자. >

![StableDiffusion_1](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/bc12c5bc-24e5-4f21-ba95-75c89648fa30){: width="600px"}
<br/>

Image Generation은 일반적으로 computational cost가 굉장이 높은 task이다. 특히 high-resolution synthesis는 autoregressive(AR) transformers와 같은 수십억 매개변수를 포함하는 likelihood 기반 모델이 수행하고 있다. 대조적으로 GAN은, adversarial learning을 사용하지만, 해당 절차가 복잡한 multi-modal distribution 모델링으로 쉽게 확장되지 않아 비교적 제한된 데이터에만 잘 동작하는 것으로 밝혀졌다. 최근 Diffusion model은 다양한 image synthesis task에서 인상적인 결과를 보여주고 있다. Likelihood 기반 모델이기 때문에 GAN처럼 mode collapse 및 training 불안정성을 나타내지 않으며 parameter sharing을 활용하여 AR 모델처럼 수십억개의 매개변수를 사용하지 않고도 natural 이미지의 복잡한 분포를 모델링 할 수 있다.
<br/>

![StableDiffusion_2](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/b1ff5c76-270c-4739-8d7d-97a0d10ddc98){: width="1100px"}
_< 기존의 Image Synthesis Model 비교 >_
<br/><br/><br/>

DM은 likelihood-based 모델에 속하여, 모델의 <mark style='background-color: var(--hl-green)'><span style='color: var(--text-color)'>mode-covering**</span></mark> 동작으로 인해 눈에 띄지 않는 detail을 모델링하는데 과도한 용량(컴퓨팅 리소스)을 소비하기 쉽다. 초기 denoising 단계에서 undersampling을 통해 이 문제를 해결하고자 했지만, 고차원 RGB 이미지 공간에서 반복적인 gradient computation을 수행하기 때문에  여전히 막대한 시간과 메모리가 소요된다.  저자는 접근성을 높이고 리소스 소비를 줄이기 위해, training과 sampling 모두에 대한 계산 복잡성을 줄이기 위한 방법의 중요성을 강조했다.

> **<mark style='background-color: var(--hl-green)'><span style='color: var(--text-color)'>Mode-covering vs Mode collapse**</span></mark>**
GAN의 경우, Generator가 샘플 전체의 Moda를 커버하지 않더라도, 이에 대한 panalty가 존재하지 않는다. 따라서 mode collapse가 빈번하게 발생한다.
반면 Likelihood-based 모델인 VAE, Diffusion의 경우에는 posterior z∼q(z|x)를 가지고 p(x|z)의 conditional generation에 대한 likelihood를 Maximize한다. 이 과정에서 데이터 포인트를 복원해야한다는 제약이 발생하고, 모델은 Modality를 모두 커버하는 시도가 발생하게 된다.
<br>참조: [https://revsic.github.io/blog/coverage/](https://revsic.github.io/blog/coverage/)
>

<br/><br/>![StableDiffusion_3](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/349c9c2a-42d4-4e84-a07b-3ee9359cae1c){: width="500px"}

그림 2는 학습된 모델의 **Rate-Distortion trade off 그래프**이다.  Digital 이미지의 대부분의 bit는 눈에 띄지 않는 세부 사항을 나타내고 있다. GAN과 AE 같은 모델은 Perceptual Compression, 즉 Bit rate가 비교적 큰 눈에 띄는 부분을 생성하는데 주력하고 있다. 반면 Diffusion Model은 픽셀단위로 생성하기 때문에 좀 더 Perceptual 하지 않은 부분에 집중하는 것을 볼 수 있다. DM의 이러한 특성은 세밀한 이미지 생성을 가능하게 하지만, 계산 요구사항이 지나치게 높아진다. 논문의 저자는 high-frequency 세부 정보를 제거하지만 의미론적 변화는 거의 없는 압축 단계를 거쳐, perceptual 측면에서는 거의 동일하지만 계산적으로 더 적합한 지점을 찾는 것을 목표로 한다.
<br/><br/>

Stable Diffuion의 단계는 크게 두단계로 나눠진다.

1. **<mark style='background-color: var(--hl-yellow)'><span style='color: var(--text-color)'>Perceptual Compression</span></mark>**: 데이터 공간과 **Perceptual 측면에서 동등한 저차원 (따라서 효율적인) 공간을 학습**하는 Auto Encoder
인코딩 단계를 한 번만 학습하면 되므로 이를 여러 DM 학습에 재사용하거나 완전히 다른 작업에 사용할 수 있는 장점이 있다. 
2.  **<mark style='background-color: var(--hl-yellow)'><span style='color: var(--text-color)'>Semantic compression</span></mark>**: 실제 Generative model이 데이터의 **의미적, 개념적 구성을 학습**
<br/><br/>

해당 방법은 댜양한 task에서 경쟁력있는 성능을 달성할 뿐 아니라 픽셀 기반 Diffusion 접근 방식에 비해 inference cost도 절감된다. 뿐만 아니라 high-resolution 또는 megapixel image 생성에도 적용될 수 있으며, multi-modal training도 가능한 장점이 있다.
<br/><br/><br/><br/><br/>

# Method

---

Diffusion Model의 배경 지식은 해당 글에서는 다루지 않는다. Diffusion Process의 배경 지식이 필요하다면 블로그 포스팅을 참고하자.
[[Diffusion 정리]](https://lunaleee.github.io/posts/Diffusion/)

![StableDiffusion_4](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/3c25895b-ada2-4df7-bf99-1440d7971cc5){: width="600px"}

이전에는 Diffusion model의 계산 복잡성을 줄이기 위해 Loss term을 적게 sampling하는 방법을 사용했지만, 이미지 pixel을 직접 예측하는 방법은 여전히 계산 비용이 큰 문제가 있다.

본 논문에서는 고해상도 이미지 합성을 위해 **<mark style='background-color: var(--hl-yellow)'><span style='color: var(--text-color)'>이미지 압축 단계</span></mark>**를 generative learning 단계에서 분리했다. Autoencoding model을 사용하여 이미지와 지각적으로 동일한 공간을 학습하지만 계산 복잡성은 줄어든다. 해당 방법은 다음과 같은 장점을 가지고 있다.

1. sampling이 저차원 공간에서 수행되기 때문에 계산 효율적이다.
2. UNet 구조에서 학습된 inductive bias를 활용하여 데이터의 spatial structure(공간 구조)를 학습하는데 효과적이므로 이미지 품질을 저하시키는 압축을 완화할 수 있다.
3. Latent space를 활용하여 여러 모델을 학습할 수 있고 down stream task에도 활용할 수 있다.
<br/><br/><br/><br/><br/>

## 1. Perceptual Image Compression

논문에서는 perceptual(지각) 압축 모델로 **VQ-GAN**을 사용한다([MaskGIT 리뷰](https://lunaleee.github.io/posts/MaskGIT/)에서도 살짝 다뤘었다). Perceptual loss와 patch 기반 adversarial objective의 조합으로 학습된 Auto encoder로 구성된다. 
<details>
  <summary><b>VQ-GAN</b></summary>
  <div markdown="1">
  ![StableDiffusion_5](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/ec3d1343-c945-443d-aed6-da8de64a26e4){: width="700px"}
  <br/>
  VQ-GAN은 CNN으로 Encoder를 사용하여 Locality 를 잘 반영하는 codebook을 학습하고, Transformer의 풍부한 표현력으로 Image Synthesis를 수행한다.
  <br/><br/>
  VQ-GAN에서는 이미지를 encoder로 압축하여 codebook 내의 code로 변환하여줌으로서 discretize 한다. 물론 이 과정에서 continuous한 feature를 우리가 가지고 있는 정보(code)만으로 표현하기 때문에 information loss가 발생할 수 있다(디테일 삭제, 왜곡 발생 등).
  ![StableDiffusion_6](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/fe7a735d-d8f6-4231-ab3f-f504d6ee7c79){: width="800px"}
  Loss function은 다음과 같이 구성된다. 
  <br/>
  VQ-GAN에서는 이전 논문인 VQ-VAE와 다르게 reconstruction loss로 **Perceptual Loss**를 사용한다. MSE는 평균 제곱오차로, 샘플이 픽셀별로 평균에서 크게 벗어나지 않도록하는 것이 목적이므로 구조적인 부분은 유지하되 이미지가 blury한 문제가 있다. 따라서 VQ-GAN에서는 이 Loss term을 perceptual loss로 변경하고 local realism을 강화하였다. 
  <br/><br/>
  Perceptual loss는 VGG 16 과 같은 feature extractor를 따로 사용하여 중간 layer에서 feature를 추출한 뒤(원본 이미지 $x$, 생성 이미지 $\hat x$) feature map 사이에서 loss를 구하는 방법이다. feature사이의 loss를 구함으로 지역적인 특성을 고려할 수 있는 장점이 있다.
  <br/><br/>
  Discriminator 학습을 위해서는 **Patch-wise Adversarial Loss**를 적용한다.
  <hr style="border: solid 0.5px lightgrey;">
  </div>
</details>
<br/>

![StableDiffusion_7](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/c562f24d-888a-4c20-8a8d-e2b605030a26){: width="700px"}

자세한 과정은 위의 그림과 같이 진행된다. 이 때 downsampling factor $f=2^m$ 으로, 다양하게 실험했다고 한다.
<br/><br/>

Latent space의 분산이 커지는 것을 막기 위해서 두가지 **Regularization**을 수행했다. 첫 번째 변형은 **KL-reg.**으로, VAE와 유사하게 Latent variable이 표준 정규 분포여야한다는 KL-penalty를 주는 것이다. 두 번째 변형은 **VQ-reg.**으로, 디코더 안에 vector quantization을 사용하는 것이다.
<br/><br/>

뒤따르는 Diffusion model은 latent space z = E(x)의 2차원 구조에서 작동하도록 설계되었기 때문에,  기존에 latent space를 임의의 1D ordering으로 압축하여 z의 고유한 구조를 무시했던 이전 연구들과 달리 세부사항을 더 잘 보존한다.
<br/><br/><br/><br/><br/>

## 2. Latent Diffusion Models

![StableDiffusion_8](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/9f4e1356-088d-480a-8d3b-1ff323f9aee1){: width="600px"}

Diffusion model은 noise 이미지에서 **<mark style='background-color: var(--hl-yellow)'><span style='color: var(--text-color)'>정규 분포 변수를 점진적으로 denoising</span></mark>** 하여 데이터 분포 p(x)를 학습하도록 설계된 모델이다. 이 과정은 길이 T의 고정된 Markov chain의 reverse process 학습에 해당한다. (해당 과정에 대한 내용은 아래 게시물을 참고)
<br/><br/><br/><br/>

![StableDiffusion_9](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/8c82e0b1-4778-4ff0-9ebe-696ec2511ecc){: width="900px"}
<br/><br/>

LDM Architecture, Pixel space라고 되어있는 분홍색 박스 부분이 압축 모델, 가운데  Latent Space라고 되어 있는 초록색 박스 부분이 Diffusion 모델, 가장 왼쪽의 Conditioning 이렇게 크게 세 개의 구조로 나뉘는 것을 볼 수 있다.

Perceptual 압축 모델($\mathcal{E, D}$)을 통해 효율적인 저차원 Latent space를 사용할 수 있게 되었다. 고차원 pixel space와 비교하여 해당 공간은 (i)데이터에서 중요한, **의미가 있는 bit에 집중**하고 (ii)**더 낮은 차원에서 많은 계산**을 효과적으로 수행할 수 있기 때문에 lilkelihood 기반 generation 모델에 적합하다. 
<br/><br/>

이전의 작업들은 고도로 압축된 discrete latent space에서 transformer 모델을 사용했던 것과는 달리, 본 논문에서는 **<mark style='background-color: var(--hl-green)'><span style='color: var(--text-color)'>image-specific inductive bias</span></mark>**를 활용할 수 있다. 그 이유는 아래와 같다.

1. **2D convolution 기반 UNet 구조**를 활용한다.
2. Reweighted bound를 사용하여 Objective function을 지각적으로 더 연관있는 bit에 집중하도록 한다.
(Diffusion model과 비교하여보면 수식은 아래와 같은 차이가 있다.)
    
    ![StableDiffusion_10](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/7ce39cda-40dd-4997-b1f7-83e000e71e3c){: width="600px"}
    
<br/><br/>
LDM 모델의 neural  backbone $\epsilon_{\theta}(\circ, t)$ 은 **<mark style='background-color: var(--hl-yellow)'><span style='color: var(--text-color)'>time-conditional U-Net</span></mark>**으로 구현된다(위의 architecture 그림 참조). Forward process는 고정되어 있기 때문에, $z_t$는 학습 도중  $\mathcal{E}$를 통해 얻을 수 있고, $p(z)$의 샘플을 $\mathcal{D}$에 통과시켜 image space로 디코딩할 수 있다.
<br/><br/><br/><br/><br/>

## 3. Conditioning Mechanisms

LDM의 Conditional denoising autoencoder $\epsilon_\theta(z_\theta, t, y)$를 이용하여 $p(z|y)$의 조건부 확률을 모델링 할 수 있다. 텍스트, semantic map과 같은 입력 y를 통해 합성 process를 제어하는 방법을 제시한다. <br/><br/>
본 논문에서는 LDM을 보다 유연한 conditional image generator로 바꾸기 위하여, 다양한 input modality에 효과적으로 동작하는 **cross-attention mechanism**을 UNet에 적용한다. 
<br/><br/><br/>

먼저, 다양한 modality의 y를 pre-process하기 위하여 domain specific encoder $\tau_\theta$를 추가한다. 이 encoder는 y를 중간 representation $\tau_\theta(y) \in ℝ^{M\times d}$로 투영하고, cross-attention layer를 사용하여 UNet의 중간 레이어에 매핑한다. Cross-attention layer는 아래와 같이 구현된다.

![StableDiffusion_11](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/d7ec5b75-7449-4fad-8a68-8f108b70cc38){: width="600px"}
<br/><br/>

여기서 $\varphi_i(z_t) \in ℝ^{N\times d^i_\epsilon}$ 은 UNet$( \epsilon_\theta)$의 (flattened) 중간 representation을 나타낸다. <br>

$W_V^{(i)} \in ℝ^{d\times d^i_\epsilon}, W_Q^{(i)} \in ℝ^{d\times d_\tau}, W_K^{(i)} \in ℝ^{d\times d_\tau}$ 는 학습가능한 projection metrics를 의미한다. 

![StableDiffusion_12](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/fbc69e3f-c2bb-458f-a402-ea44300ae8a3){: width="700px"}
<br/><br/>

Image-conditioning pairs를 기반으로 아래의 수식을 통해 조건부 LDM을 학습한다.

![StableDiffusion_13](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/ea978212-88d9-4e57-8612-f8251a3dfcee){: width="600px"}

여기서 $\tau_\theta$ 와 $\epsilon_\theta$는 위의 수식을 통해 공동으로 학습된다. 해당 conditioning mechanism은 각각의 domain-specific한 모델을 붙여 사용할 수 있으므로, 유연한 구조를 가진다.
<br/><br/><br/><br/><br/>

# Experiments

---

LDM은 다양한 이미지 합성 방법을 제공하지만, 그 전에 Pixel 기반 Diffusion 모델과의 비교하여 모델의 이점을 분석하였다. VQ regularized latent spaces에서 학습된 LDM은 reconstruction 성능이 다른 모델보다 떨어지더라도 생성 이미지의 품질이 다른 모델들과 비교하여 우수하다는 것을 발견했다고 한다. 
<br/><br/><br/>

### 1. On Perceptual Compression Tradeoffs

이 섹션에서는 압축 모델의 다양한 downsampling factor ($f=2^m$ ) $f \in$ {1, 2, 4, 8, 1, 32} (LDM-f라고 정함)에 대한 LDM의 동작에 대해 조사했다. 

![StableDiffusion_14](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/75e0b0e0-24e9-497e-8c02-d69256774c9e){: width="800px"}

위의 그래프는 ImageNet으로 class 조건부 모델을 2M step 학습할 때 샘플 품질에 대한 그래프이다. 위의 그래프를 통해 downsampling factor가 너무 작으면 학습이 느려지고, f 값이 너무 크면 비교적 적은 학습 단계 후에 이미지 품질이 정체되는 것을 알 수 있다. 
<br/><br/><br/><br/>

### 2. Image Generation with Latent Diffusion

해당 섹션에서는 CelebA-HQ, FFHQ, LSUN-Churches, LSUN-Bedrooms 데이터셋을 활용하여 unconditional model을 학습시켜 생성 이미지 샘플 품질과 data manifold에 대한 coverage를 조사했다. 

![StableDiffusion_15](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/11d65983-e1bb-4339-8796-2a5c2a2d3b68){: width="600px"}

![StableDiffusion_16](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/ab073e82-bac2-4a09-803f-00b73ca1f734){: width="1100px"}
<br/><br/><br/><br/>

### 3. Conditional Latent Diffusion

다음은 LAION 데이터셋에 대해 학습한 Text-to-Image task 결과 이미지이다.

![StableDiffusion_17](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/3c987500-621b-4f21-9de9-6069d303d61a){: width="1100px"}
<br/><br/>

추가적으로 메가 픽셀 이미지에 대한 semantic synthesis 작업에 LDM을 적용했다. $256^2$으로 학습된 모델을 사용하여 고화질 이미지(512 X 1024)를 생성하였다. 

![StableDiffusion_18](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/4e8b826f-7417-4fc5-801d-7bdfc5cc8c44){: width="600px"}
<br/><br/><br/><br/>

### 4. Super-Resolution with Latent Diffusion

ImageNet-Val 데이터셋에 대하여  64→256 super-resolution을 수행하였다. 

![StableDiffusion_19](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/ac5cf322-796d-4d6e-90eb-b9dbf2d78275){: width="500px"}
<br/><br/><br/><br/>

### 5. Inpainting with Latent Diffusion

Inpatinting은 이미지의 일부가 손상되었거나 이미지 내에 존재하는 원하지 않는 컨텐츠를 대체하여 새로운 컨텐츠로 채우는 작업을 의미한다. 원하지 않는 부분을 mask로 설정하여해당 부분을 채우도록 학습할 수 있다. 아래 그림은 Inpainting 학습에 대한 결과 이미지이다.

![StableDiffusion_20](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/2c51bc21-051a-452d-977f-96f521d53823){: width="500px"}
