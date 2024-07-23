---
title: "[논문 리뷰] Prompt-to-Prompt Image Editing with Cross Attention Control"
author: lunalee
date: 2024-06-21 21:19:35 +0900
categories: [AI, Paper Review]
tags: [Diffusion, Generation]
pin: false
math: true
---

<br/><br/>
`Google Research` `arXiv 2022`

- Paper: [https://arxiv.org/abs/2208.01626](https://arxiv.org/abs/2208.01626)
- Git: [https://github.com/google/prompt-to-prompt/](https://github.com/google/prompt-to-prompt/)
- Page: [https://prompt-to-prompt.github.io](https://prompt-to-prompt.github.io/)
<br/><br/><br/>

#### 📖 핵심 훑어보기 !!

- Prompt 조작만으로 **생성된 이미지에서 원래 구조를 유지**하면서 편집할 수 있는 textual editing 방법 제안
- 생성된 이미지의 구조와 모양은 **Diffusion process 과정에서 pixel과 text embedding 간의 interaction에 의존**한다는 사실에 기반하여, **Cross-attention layer에서 발생하는 pixel-to-text interaction을 수정**하는 방법 제안
- 수정된 prompt에서 생성된 cross attention map에 원래 prompt에서 생성된 attention map을 **단계별로 injection** 하여 원본의 구조는 유지하면서도 수정된 prompt를 반영하는 새로운 이미지를 생성
<br/><br/><br/><br/>

# Introduction

---

LLI(large-scale language-image)는 뛰어난 image generation 성능을 보여주고 있다. 하지만 **image editing** 측면에서, 간단하게 이미지를 수정할 수 있는 수단이 없을 뿐 아니라 특정 semantic region에 대한 컨트롤이 불가능하다. Text prompt를 약간만 변경하더라도 완전히 다른 출력 이미지가 생성되기 쉽다.

기존에는 이런 문제를 해결하기 위해 사용자가 이미지에서 일부를 명시적으로 **masking**하고 해당 부분만 변경되도록 하는 방법이 제안되었다. 하지만 masking 절차가 번거롭고 중요한 구조적 정보가 제거되는 문제가 있다.
<br/><br/>

본 논문에서는 **<mark style='background-color: var(--hl-yellow)'><span style='color: var(--text-color)'>Prompt-to-Prompt 조작</span></mark>**을 통해 pre-trained text conditioned diffusion 모델에서 이미지를 편집하는 **<mark style='background-color: var(--hl-yellow)'><span style='color: var(--text-color)'>textual editing</span></mark>** 방법을 소개한다. 이를 위해 cross-attention map을 조작하여 diffusion process에 주입함으로서 이미지를 편집한다. 특히, diffusion process 중에 어떤 pixel이 prompt text의 어떤 token에 attention하는지 제어할 수 있다.
<br/><br/>

![P2P_1.png](https://github.com/user-attachments/assets/b0920e57-9d09-4202-8cff-45b5534fe36f){: width="1000px"}

위의 그림과 같이 논문에서 cross-attention을 제어하는 여러 방식이 있다.

1. prompt에서 하나의 token을 변경 (e.g. 개 → 고양이)
2. 이전의 token은 freeze하고 새로운 단어를 추가하여 전반적인 이미지 수정 (e.g. style 변경)
3. 생성된 이미지에서 단어의 의미적 효과를 증폭하거나 약화시킴
<br/><br/>

논문의 방법은 textual prompt만을 편집하므로, 빠르고 직관적인 편집이 가능함과 동시에 추가적인 학습, 데이터가 필요하지 않은 장점이 있다. 
<br/><br/><br/><br/><br/><br/>

# Method

---

먼저 notation과 목표를 다음과 같이 정의하자.

- $\mathcal{I}$ : text prompt $\mathcal P$와 random seed $s$를 사용하여 text-guided diffusion 모델을 통해 생성된 이미지
- Goal:  편집된 text prompt $\mathcal P^\ast$를 이용하여 **<mark style='background-color: var(--hl-yellow)'><span style='color: var(--text-color)'>편집된 이미지 $\mathcal I^\ast$를 생성</span></mark>**하는 것
<br/><br/>

생성된 이미지에서 **원래의 이미지의 모양과 구조를 유지**하면서 이미지의 일부분을 수정하려고 할 때, (e.g. “my new bicycle” 이라는 prompt에서 생성된 이미지에서 자전거를 스쿠터로 변경한다던지, 자전거의 색상을 변경) 생각할 수 있는 가장 단순한 방법으로 diffusion process의 **random seed를 고정**하고 **입력 prompt를 수정**하는 방법이다. 
<br/><br/>

![P2P_2.png](https://github.com/user-attachments/assets/60d4544d-b062-4ba3-a357-f11015e91ae8){: width="1200px"}

이와 같은 방법을 사용하면 위와 같이 구조와 구성이 완전히 다른 이미지가 생성되었다. 여기서 중요한 문제는 생성된 이미지의 구조와 모양은 random seed 뿐 아니라 **<mark style='background-color: var(--hl-yellow)'><span style='color: var(--text-color)'>diffusion process를 통한 pixel과 text embedding 간의 interaction에 의존</span></mark>**한다는 것이다. 따라서 **<mark style='background-color: var(--hl-green)'><span style='color: var(--text-color)'>Cross-attention layer에서 발생하는 pixel-to-text interaction을 수정</span></mark>**하여 **Prompt-to-Prompt image editing**을 수행하는 것이 본 논문의 접근 방법이다.
<br/><br/><br/><br/><br/>

## 1. Cross-attention in text-conditioned Diffusion Models

먼저, 논문에서는 backbone 모델로 Imagen을 사용하였다. 하지만 논문의 방법은 일반적인 diffusion 모델이 적용 가능하다고 한다.
<br/><br/>

> **Imagen** ([Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding📄](https://arxiv.org/abs/2205.11487) )
> 
> ![P2P_3.png](https://github.com/user-attachments/assets/1a371d99-55c7-49e4-bac4-ac7a227d74a8){: width="600px"} <br/>
>
> Imagen은 크게 세부분으로 구성된 Diffusion 모델이다. <br/>
> **1. Pre-trained Text Encoder**: text embedding 생성 <br/>
> **2. Diffusion model** (Classifier-free guidance): text embedding을 바탕으로 이미지 생성 <br/>
> **3. Cascaded Diffusion model** (Super resolution): high-resolution image로 upscale <br/>
> 

<span style='color: var(--txt-gray)'>(위 모델 구조에서 이미지의 구성, geometry 등은 64 X 64 text-to-image diffusion 모델(2번째 모델)에서 결정되므로 본 논문에서는 Diffusion 모델에만 방법을 적용하고 SR 모델은 그대로 사용하였다.)</span>
<br/><br/><br/>

일반적으로 각 diffusion step $t$에서 U-net 기반의 모델을 사용하여 noisy한 이미지 $z_t$, text embedding $\psi (\mathcal P)$로부터 noise $\epsilon$을 예측한다. 모든 step이 진행되면 마지막에서 이미지 $\mathcal I = z_0$가 생성된다. 중요한 점은 두 modality 사이의 interaction이 noise prediction 과정 중 발생한다. 이 때 **<mark style='background-color: var(--hl-green)'><span style='color: var(--text-color)'>Cross-attention layer</span></mark>**에서 visual, textual feature fusion이 발생하고 **<mark style='background-color: var(--hl-green)'><span style='color: var(--text-color)'>각 textual token에 대한 spatial attention map이 생성</span></mark>**된다.
<br/><br/><br/><br/><br/>

![P2P_4.png](https://github.com/user-attachments/assets/4045d34f-465f-4c5c-8d9e-6838f60fd303){: width="1100px"}

학습된 linear projection $\ell_Q,\ell_K, \ell_V$ 를 사용하여, 위 그림과 같이 noisy 이미지의 spatial feature $\phi (z_t)$는 query matrix $Q = \ell_Q(\phi(z_t))$로 project 되고, textual embedding은 key matrix $K = \ell_K(\psi(\mathcal P))$와 value matrix $V = \ell_V (\psi(\mathcal P))$로 project 된다. 이 때의 attention map은 다음과 같다.

$$
M = \text{Softmax} \bigg(\frac{QK^T}{\sqrt{d}}\bigg)
$$

여기서 cell $M_{ij}$는 pixel $i$와 $j$번째 token에 대한 가중치이고, $d$는 $Q$와 $K$의 latent projection dimension이다. 마지막으로 cross-attention의 출력은 $\hat \phi (z_t) = MV$ 가 되고, spatial feature $\phi (z_t)$를 업데이트 하는데 사용된다.
<br/><br/><br/>

위의 그림과 같이 직관적으로 cross-attention output $MV$는 attention map $M$을 가중치로 하는 $V$의 **weighted average**임을 알 수 있다(M은 Q와 K의 similarity와 상관관계임). 추가적으로 표현력을 높이기 위해 multi-head attention을 사용했다.
<br/><br/><br/><br/><br/>

## 2. Controlling the Cross-attention

앞에서 일반적인 cross-attention layer를 살펴봤다면, 이제 **생성된 이미지의 공간적 layout과 geometry가 cross-attention map에 달려있다**는 key point로 넘어오자. 
<br/><br/>

아래 그림은 visualization을 위해 **average attention map**을 구한 것이다. 그림을 통해 pixel과 text간의 interaction을 확인할 수 있다. Attention map은 각각 instance에 대해 분리된 형태로 유지된다(곰 ↔ 새). <br/>
이미지의 구조가 diffusion process의 초기 step에 이미 결정되는 것 또한 확인할 수 있다.

![P2P_5.png](https://github.com/user-attachments/assets/538c774d-d9cf-441a-9a17-60b81233fb50){: width="1200px"}
<br/><br/><br/>

Attention은 전체 composition을 반영하므로, original prompt $\mathcal P$로 생성한 attention map $M$을 modified prompt $\mathcal P^\ast$로 생성하는 과정에 주입할 수 있다. 이를 통해 input image $\mathcal I$의 구조를 보존하면서도 수정된 prompt를 반영하는 edited image $\mathcal I^\ast$를 합성할 수 있다. 
<br/><br/><br/>

먼저 controlled image generation을 위한 general framework를 살펴보자. 

- $DM(z_t, \mathcal P, t, s)$ : single step t의 diffusoin process 연산. noisy image z_{t−1}와 attention map M_t를 output으로 생성함.
- $DM(z_t, \mathcal P, t, s)\lbrace M \gets \widehat M \rbrace$ : 보충된 prompt의 value $V$를 유지하면서, attention map $M$을 새로운 map $\widehat M$ 으로 override하는 diffusion step.
- $M^\ast_t$ :편집된 prompt $\mathcal P^*$를 사용하여 생성된 attention map.
- $Edit(M_t, M_t^∗, t)$ : $t$번째 attention map을 입력으로 받는 general edit function.
<br/><br/><br/>

두 prompt에 대해 동시에 iterative diffusion process를 진행한다. 이 때 같은 prompt라도 random seed가 다른 경우 완전히 다른 출력이 생성되는 diffusion 모델의 특성을 고려하여, randomness를 고정했다. General algorithm은 아래와 같이 진행된다.

![P2P_6.png](https://github.com/user-attachments/assets/42e997b1-fe52-48dc-a8a4-596b0efe3921){: width="1200px"}

- (line 3,4): random seed s로 Gaussian random variable $z_T (= z^*_T)$생성
- (line 6,7): 원본 prompt와 편집된 prompt를 사용하여 random variable로부터 각각 diffusion process 진행, attention map $M_t, M^*_t$ 생성
- (line 8): 원본 prompt로 생성된 attention map $M_t$와 수정된 prompt로 생성된 $M^*_t$를 사용하여 $Edit(\cdot)$과정을 거쳐 수정된 attention map $\widehat M_t$ 생성
- (line 9): 수정된 attention map $\widehat M_t$를 이용하여 $z^*_{t-1}$ 생성
<br/><br/>

다음으로는 비어있는 $Edit(M_t,M_t^∗,t)$ 부분을 정의하기 위한 3가지 specific editing operation에 대해 알아보자. 
<br/><br/><br/><br/><br/>

#### 1. Word Swap.

Word swap은 말 그대로 **<mark style='background-color: var(--hl-yellow)'><span style='color: var(--text-color)'>원래 prompt의 token을 다른 token과 바꾸는 것</span></mark>**을 말한다(e.g. $\mathcal P$ = "a big red **bicycle**"에서 $\mathcal P^∗$ = "a big red **car**").

원래의 구성을 보존하는 동시에 새 prompt의 내용을 처리하기 위해, 수정된 prompt로 이미지를 생성할 때 **원래 이미지의 attention map을 주입**한다. 그러나 “bicycle”에서 “car”로의 변경과 같이 큰 구조적 변경이 필요한 경우 geometry를 과도하게 제약하지 않도록 하기 위해 **<mark style='background-color: var(--hl-green)'><span style='color: var(--text-color)'>softer attention constrain</span></mark>**을 사용했다. 

$$
Edit(M_t, M^*_t, t):= \begin{cases} 
M^*_t \qquad  \text{if} \; t < \tau \\ M_t \qquad \text{otherwise.}
\end{cases}
$$

여기서 $\tau$는 injection이 적용되는 step을 결정하는 timestamp parameter이다. 앞서 언급했듯 구성은 diffusion process 초기에 결정되므로, injection step의 수를 제한함으로써 새로운 prompt에 적응하는데 필요한 **geometry freedom**을 허용할 수 있다.
<br/><br/><br/><br/><br/>

#### 2. Adding a New Phrase.

![P2P_7.png](https://github.com/user-attachments/assets/88cf50ea-50d8-497b-a94d-edbf8ede365b){: width="900px"}

다음으로는 prompt에 **새 token을 추가**하는 방법을 살펴보자($\mathcal P$ = "a castle next to a river"에서 $\mathcal P^∗$ = "**children drawing of** a castle next to a river") 

공통적인 detail을 보존하기 위해 **<mark style='background-color: var(--hl-yellow)'><span style='color: var(--text-color)'>두 prompt의 공통된 token에만 attention injection을 적용</span></mark>**한다. 먼저 **<mark style='background-color: var(--hl-green)'><span style='color: var(--text-color)'>alignment function $A$</span></mark>**를 사용한다. 이 function은 target prompt $\mathcal P^*$의 token index를 입력으로 받아 **원래 prompt $\mathcal P$에서 대응하는 token index**(또는 $None$)를 출력한다. 

$$
Edit(M_t, M^*_t, t):= \begin{cases} 
(M^*_t)_{i,j} \quad \qquad  \text{if} \; A(j) = None \\ (M_t)_{i, A(j)} \qquad \text{otherwise.}
\end{cases}
$$

이 때 $None$이면(즉, 대응하는 부분이 없으면) $M^*_t$를 출력하고 아니면(대응하는 부분이 있으면) $M_t$를 출력한다. 식에서 index $i$는 pixel value, $j$는 이에 대응하는 text token을 나타낸다.
<br/><br/><br/><br/><br/>

#### 3. Attention Re–weighting.

마지막으로 **각 token이 미치는 영향을 강화하거나 약화**시키고자 하는 경우를 살펴보자(e.g. $\mathcal P$ = “a **fluffy** red ball”을 더, 혹은 덜 fluffy하게).

이 경우에는 parameter $c \in [−2, 2]$ 를 사용하여 **<mark style='background-color: var(--hl-yellow)'><span style='color: var(--text-color)'>특정 token $j^*$의 attention map을 scale하여 조절</span></mark>**하는 방법을 사용한다. 나머지 attention map은 변경되지 않는다.

$$
(Edit(M_t, M^*_t, t))_{i,j}:= \begin{cases} 
c \cdot (M_t)_{i,j} \qquad  \text{if} \; j = j^* \\ (M_t)_{i,j} \qquad \quad \text{otherwise.}
\end{cases}
$$

<br/><br/><br/><br/><br/>

# Experiments

---

### 1. Text-Only Localized Editing.

![P2P_8.png](https://github.com/user-attachments/assets/72b31655-9229-4da6-9c88-3e17021acfcd){: width="800px"}

먼저 local 편집에 대한 결과이다. 그림의 윗부분과 같이 배경이 잘 보존됨과 동시에 수정된 prompt를 잘 반영하는 것을 볼 수 있다. 반면에 논문의 방법을 사용하지 않고 단순하게 random seed만을 고정한 아래 부분의 결과는 완전히 다른 geometry의 이미지를 생성한 것을 볼 수 있다. 
<br/><br/><br/>

![P2P_9.png](https://github.com/user-attachments/assets/4372bc22-cf75-4b27-9fe7-24234f7c18c6){: width="800px"}

또한 texture 편집 뿐 아니라 구조적인 수정을 수행할 수 있다. 위의 그림과 같이 cross attention injection을 적용하는 diffusion step을 변경하여 원본 이미지에 대한 충실도를 제어할 수 있다. Injection을 수행하는 stepdl 많을수록 원래 이미지에 대한 충실도가 높아진다.
<br/><br/><br/><br/>

### 2. Global editing.

![P2P_10.png](https://github.com/user-attachments/assets/7a103923-4db1-4529-a1d8-adf5e9d0e6a2){: width="800px"}

기존의 prompt에 새로운 단어를 추가하여 위 그림처럼 배경은 그대로 유지하면서 원본 이미지에 대한 추가 세부 정보를 생성할 수 있다.

뿐만 아니라 아래부분 이미지처럼 Global한 부분을 변경하면서도 원래의 이미지 content를 유지할 수 있다.
또한 texture 편집 뿐 아니라 구조적인 수정을 수행할 수 있다. 위의 그림과 같이 cross attention injection을 적용하는 diffusion step을 변경하여 원본 이미지에 대한 충실도를 제어할 수 있다. Injection을 수행하는 stepdl 많을수록 원래 이미지에 대한 충실도가 높아진다.
<br/><br/><br/><br/>

### 3. Fader Control using Attention Re-weighting.

![P2P_11.png](https://github.com/user-attachments/assets/460e8fb2-41e6-461e-bf4d-6110d551b24e){: width="800px"}

Prompt를 편집하여 이미지를 제어할 수 있지만 단어의 정도를 제어하기는 어렵다. 예를 들어 “snowy mountain”에서 눈 덮인 정도를 제어하고 싶을 수도 있다. 이를 위해 저자는 특정 단어로 유도되는 효과의 크기를 제어하는 fader control을 제안했다. 지정된 단어의 attention을 re-scaling하여 이러한 제어를 수행했다.
또한 texture 편집 뿐 아니라 구조적인 수정을 수행할 수 있다. 위의 그림과 같이 cross attention injection을 적용하는 diffusion step을 변경하여 원본 이미지에 대한 충실도를 제어할 수 있다. Injection을 수행하는 stepdl 많을수록 원래 이미지에 대한 충실도가 높아진다.
<br/><br/><br/><br/>

### 4. Real Image Editing.

![P2P_12.png](https://github.com/user-attachments/assets/eb90091a-3b98-4542-9936-d306c6827c24){: width="1000px"}

실제 이미지를 편집하기 위해서는 diffusion process에 입력되면 해당 이미지를 생성하는 초기 noise vector를 찾아야한다. 이를 위해 inversion이라고 알려진 process를 적용했다. 해당 process는 현재 text-guided diffusion 모델에서의 연구는 부족하기 때문에 충분히 정확하지는 않지만, 위 그림처럼 만족스러운 결과를 생성하는 것을 볼 수 있다.
<br/><br/><br/><br/>
