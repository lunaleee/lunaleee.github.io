---
title: "[논문 리뷰] ControlNet, Adding Conditional Control to Text-to-Image Diffusion Models"
author: lunalee
date: 2024-06-13 20:12:49 +0900
categories: [AI, Paper Review]
tags: [Multi-modal, Diffusion, Generation]
pin: false
math: true
---

<br/><br/>
`ICCV 2023`

- Paper: [https://arxiv.org/abs/2302.05543](https://arxiv.org/abs/2302.05543)
- Git: [https://github.com/lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet)
<br/><br/><br/>

#### 📖 핵심 훑어보기 !!

- Text-to-image 생성에서 복잡한 레이아웃이나 포즈, 모양을 제어하기 위해 Canny edges, key point, segmentation map 등 **다양한 입력**을 사용하여 diffusion model을 제어하는 ControlNet 구조 제안.
- 대규모 **pre-trained diffusion 모델의 parameter를 고정**하고, **“trainable copy”**라는 추가적인 block을 사용해 large 모델의 quality나 능력은 보존하면서도 대규모 pre-trained 모델을 재사용하는 효과를 얻음.
- ControlNet의 trainable copy에서 **zero convolution**을 사용하여 학습 초기에 random noise로부터 backbone을 보호한다.
<br/><br/><br/><br/>

# Introduction

---

Stable Diffusion과 같은 text-to-image diffusion 모델의 등장으로 text prompt를 통한 다양한 이미지 생성이 가능해졌다. 이러한 방법들은 사실적이고 높은 퀄리티의 이미지를 제공하지만, 공간적인 구성에 대한 제어가 제한적이다. 복잡한 레이아웃이나 포즈, 모양에 대해 텍스트만으로 원하는 이미지와 정확하게 일치하는 이미지를 생성하는 것은 어려운 문제이다. 
<br/><br/>

이러한 문제를 극복하기 위해 저자는 대규모의 pre-train된 **<mark style='background-color: var(--hl-yellow)'><span style='color: var(--text-color)'>Stable diffusion에 대한 조건 제어를 학습하는 end-to-end 구조</span></mark>**인  **<mark style='background-color: var(--hl-yellow)'><span style='color: var(--text-color)'>ControlNet</span></mark>**을 제시했다. 해당 논문에서는 대규모 pre-trained diffusion 모델을 backbone으로 사용한다. Large diffusion 모델의 parameter를 고정하고, ControlNet의 encoding layer에서 trainable copy를 만드는 방식을 사용해 large 모델의 quality나 능력은 보존할 수 있다. 이 때 zero convolution layer를 도입하여 학습이 안정적으로 진행될 수 있도록 했다.
<br/><br/>

![ControlNet_1.png](https://github.com/user-attachments/assets/7d75cd0c-5c28-4e51-b954-9d3434b9b2c5){: width="1000px"}

논문에서는 ControlNet을 사용하여 Canny edges, Hough lines, human key point, segmentation map, depth 등 다양한 컨디셔닝 입력을 사용하여 stable diffusion을 제어했다. 위의 그림과 같이 다양한 입력에 대해 안정적으로 결과 이미지가 생성되는 것을 볼 수 있다. 구체적인 방법에 대해 알아보자.
<br/><br/><br/><br/><br/><br/>

# Method

---

## 1. ControlNet

![ControlNet_2.png](https://github.com/user-attachments/assets/37a28738-fa7d-4602-838a-1e8332d49767){: width="600px"}

ControlNet은 위의 그림과 같이 neural network block에 condition을 주입한다. 여기서 network block이라는 단어는 ResNet block, Trasformer block 등과 같은 일련의 neural layer를 나타낸다.
<br/><br/><br/>

- $\mathcal F(\cdot \ ; \Theta)$는 parameter $\Theta$를 갖는 학습된 neural block이고 입력 feature map $x$를 $y$로 변환한다($x \in ℝ^{h \times w \times d}$, $h$: height, $w$: width, $d$: depth).
    
    $$
    y = \mathcal F(x \ ; \Theta)
    $$
    
- Pre-trained neural block에 ControlNet을 추가하기 위해 block의 parameter  $\Theta$를 고정(freeze)하고, parameter $\Theta_c$를 갖는 **<mark style='background-color: var(--hl-green)'><span style='color: var(--text-color)'>trainable copy</span></mark>**로 block을 복제한다(그림 (b)).
- trainable copy는 external conditioning vector $c$를 입력으로 받는다.
- trainable copy는 zero convolution **layer $Z(\cdot \ ; \cdot)$를 사용한다. zero convolution은 weight와 bias가 모두 0으로 초기화된 $1 × 1$ convolution layer이다.
- 2개의 **<mark style='background-color: var(--hl-yellow)'><span style='color: var(--text-color)'>zero convolution</span></mark>**을 사용하는데, 각각 paremeter $\Theta_{z1}, \Theta_{z2}$를 가질 때 ControlNet은 다음과 같이 계산된다. 
(여기서 $y_c$는 ControlNet의 출력이다)
    
    $$
    y_c = \mathcal F(x \ ; \Theta) \ + \ \mathcal Z(\mathcal F(x + \mathcal Z(c \ ; \Theta_{z1}); \Theta_c); \Theta_{z2})
    $$
    
- 첫 번째 학습 단계에서 zero convolution이 모두 0으로 초기화되므로 위 식의 $Z(\cdot \ ; \cdot)$항은 모두 0이 된다. 따라서 <mark style='background-color: var(--hl-green)'><span style='color: var(--text-color)'>$y_c=y.$</span></mark> 즉, ControlNet의 출력은 pre-trained 모델의 출력과 같아지게 된다. 결과적으로 학습이 시작될 때 유해한 noise가 trainable copy의 hidden layer에 영향을 미칠 수 없다.
<br/><br/><br/><br/>

이 구조가 Stable Diffusion과 같은 대규모 모델에 적용되었을 때, 고정된 parameter로 인해 수십억 개의 이미지로 학습된 기존 모델을 보존하는 반면, trainble copy는 이러한 대규모 pre-trained 모델을 재사용하여 다양한 입력 조건을 처리하기 위한 강력한 backbone을 구축할 수 있다.
<br/><br/>

특히 zero convolution을 사용하여 학습 초기에는 trainable copy가 기능적으로 large, pretrained 모델의 역할을 수행하므로, 강력한 backbone 역할을 할 수 있다. 초기 학습 단계에서 gradient의 random noise를 제거하여 backbone을 보호한다. 
<br/><br/><br/><br/><br/>

## 2. ControlNet for Text-to-Image Diffusion

![ControlNet_3.png](https://github.com/user-attachments/assets/c6ea81e7-1859-42f1-a8e7-3486a60bfe11){: width="500px"}

Stable Diffusion의 U-net 구조에 ControlNet을 추가한 구조는 위 그림과 같다. Encoder, middle block, skip-connected decoder로 이루어진 U-net의 Encoder level에 ControlNet이 적용된다. 구체적으로 총 12개의 encoding block과 1개의 middle block의 trainable copy가 생성된다. Output은 12개의 skip-connection과 1개의 middle block에 더해진다(그림(b) 참조). 

Text Encoder로는 CLIP을 사용했다. Stable Diffusion은 일반적인 U-net 구조이므로 다른 모델에도 쉽게 적용이 가능하다. 
<br/><br/>

Stable Diffusion은 VQ-GAN과 유사한 전처리 방법을 사용하여 512×512 pixel space 이미지를 64×64 latent 이미지로 변환한다. 따라서 ControlNet에서도 Stable Diffusion과 맞추기 위해 64×64 feature space vector로 변환해야한다. 이를 위해 tiny network $\mathcal E(\cdot)$를 사용하여 image-space condition $c_{\text i}$를 feature space conditioning vector $c_{\text f}$로 인코딩했다.

$$
c_{\text f} = \mathcal E(c_{\text i})
$$

<br/><br/><br/><br/>

## 3. Training

학습 과정은 일반적인 Diffusion 모델의 denoising process와 동일하다. 이 부분은 [Stable Diffusion 논문 리뷰](https://lunaleee.github.io/posts/stablediffusion/)를 살펴보자.

학습 과정에서 text prompt $c_t$의 50%를 빈 문자열로 무작위로 바꿨다. 이 방식은 입력 conditioning 이미지(예: edges, poses, depth 등)에서 의미를 직접 인식하는 ControlNet의 능력을 향상시킨다. 
<br/><br/>

![ControlNet_4.png](https://github.com/user-attachments/assets/10e9af08-217c-41dc-9a2c-4e0ecc00ea45){: width="500px"}

논문의 방식은 모델이 control condition을 점진적으로 학습하지 않고 위의 그림의 6133 step에서와 같이 갑자기 conditioning 이미지를 따라가는 현상이 발생한다. 이를 **“sudden convergence phenomenon”**이라고 지칭했다.
<br/><br/><br/><br/><br/>

## 4. Inference

![ControlNet_5.png](https://github.com/user-attachments/assets/970f46ab-e02d-41e1-8ee5-fbe692c78ea8){: width="500px"}

입력으로 여러개의 conditioning 이미지(e.g. Canny edge, pose)를 적용하기 위해 ControlNet의 해당 output을 바로 Stable Diffusion에 추가해줄 수 있다. 추가적인 weight이나 linear interpolation없이 위의 그림과 같이 여러개의 condition을 충족하는 이미지를 생성하는 것을 볼 수 있다.
<br/><br/><br/><br/><br/><br/>

# Experiments

---

### 1.  Qualitative Results

![ControlNet_6.png](https://github.com/user-attachments/assets/fbc3558a-7028-418c-958b-752f30ebad22){: width="800px"}

위의 그림은 prompt 없이 다양한 condition에 생성된 결과이다.
<br/><br/><br/><br/><br/>

### 2. Ablative Study

본 논문의 구조에 대한 성능 평가를 위해 아래의 두가지 조건으로 구조를 변경하여 실험을 진행했다.

1.  zero convolution을 gaussian weight로 초기화된 standard convolution layer로 대체
2. 각 block의 trainable copy를 ControlNet-lite라고 하는 단일 convolution layer로 대체
<br/><br/><br/>

또한 4가지 prompt 설정에 대해 실험했다.

1. prompt 없음
2. conditioning 이미지의 객체를 완전히 포함하지 않는 불충분한 prompt (e.g. “a high-quality, detailed, and professional image”)
3. conditioning 이미지의 의미와 상충되는 prompt
4. 필요한 content 의미를 설명하는 완벽한 prompt (e.g. "a nice house")
<br/><br/>

아래의 그림은 4가지 prompt 설정에 대해 모두 성공적인 이미지를 생성하는 것을 보여준다. 

![ControlNet_7.png](https://github.com/user-attachments/assets/48f9d99e-8966-4732-b020-9e103330d171){: width="900px"}
<br/><br/><br/><br/><br/>

### 3. Quantitative Evaluation

![ControlNet_8.png](https://github.com/user-attachments/assets/b4d9f829-5fe9-4a46-895f-59c006f595c7){: width="400px"}

20개의 손으로 그린 스케치에 대해 5개의 방법으로 이미지를 생성했다. 12명의 사용자에게 "표시된 이미지의 품질"과 "스케치의 충실도" 측면에서 개별적으로 순위를 매기도록 요청했다.  사용자가 각 결과를 1~5점 척도로 평가하는 Average Human Ranking(AHR)를 사용했다. 결과는 위 표와 같다.
<br/><br/><br/>

![ControlNet_9.png](https://github.com/user-attachments/assets/86d00ddf-3cf5-465c-a7f2-6199459542c0){: width="500px"}

![ControlNet_10.png](https://github.com/user-attachments/assets/2c4c10b0-38f0-47b6-9e85-e895319b33c3){: width="500px"}

ADE20K의 testset을 사용하여 conditioning 충실도를 평가했다. SOTA segmentation method인 OneFormer를 사용하여 생성된 이미지에 대해 segmentation을 수행하고 IoU를 계산했다.

FID를 사용하여 distribution distance를 측정했다. 또한 text-image CLIP score와 CLIP aesthetic score를 측정했다. 결과는 위 두 개 표와 같다.
<br/><br/><br/><br/><br/>

### 4. Comparison to Previous Methods

![ControlNet_11.png](https://github.com/user-attachments/assets/2be8878c-faa5-459c-859e-20470753da01){: width="450px"}

위의 그림은 baseline과 논문의 방법(Stable Diffusion + ControlNet)의 시각적 비교를 보여준다. ControlNet은 다양한 conditioning 이미지를 견고하게 처리하고 선명하고 깨끗한 결과를 생성한다.
<br/><br/><br/><br/>
