---
title: "Generative model 기초 3. Diffusion 정리"
author: lunalee
date: 2024-01-30 19:12:42 +0700
categories: [AI, Study]
tags: [Image, Generation, Diffusion]
pin: false
math: true
---

<br/>
Diffusion model은 VAE, GAN과 같은 Generative model(생성 모델)의 일종으로, 기존의 생성 모델에 비해 안정적이고 뛰어난 성능을 보여주고 있다. GAN은 adversarial training 방식으로 인해 이미지 생성 diversity가 떨어질 뿐 아니라 mode collapse 문제 등 불안정한 특성을 가지고 있다. VAE도 latent variable z에서 변환하는 특성으로 인해 diversity가 떨어지고 surrogate loss(대리 손실)을 사용한다는 문제가 있다.

![Diffusion_1](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/dc9f48d6-2b91-4a33-9583-d2af4b494a8c){: width="px"}
<br/><br/><br/>

 Diffusion은 말 그대로 확산, Langevin dynamics(랑주뱅 동역학)라는 열역학에서 영감을 받은 방법으로 분자가 확산 현상을 통해 퍼지는 것을 역추적하여 움직임을 계산해내는 것과 비슷한 방법이다. 원본 데이터에 Random noise를 단계별로 추가하는 Markov chain을 정의한 뒤, 이 방법을 reverse하여 Noise로 부터 원하는 이미지를 생성하는 방법을 학습한다. 이 글에서 Diffusion model의 원리 및 학습 방법을 공부해보자.
<br/><br/><br/><br/>

# Diffusion Process

---

기존에 GAN과 같은 모델은 Noise(정규분포) image에서 모델을 거쳐 이미지를 생성한다. 

Diffusion model은 <mark style='background-color: var(--hl-yellow)'><span style='color: var(--text-color)'>입력 이미지에 여러 단계에 걸쳐 Noise를 더하여 입력 이미지를 Noise image로 만들고(forward process)</span></mark>, 다시 이 <mark style='background-color: var(--hl-yellow)'><span style='color: var(--text-color)'>Noise image를 여러 단계에 걸쳐 Noise를 제거함으로서 이미지를 생성(reverse process)</span></mark>한다.

Noise를 단계별로 추가하면 현재 step과 다음 step을 비교하여 어느 부분에 noise가 추가되었는지 알 수 있듯, 반대로 현재 step에서 이전 step으로 noise를 제거하는 것이 가능하다. 이렇게 단계별로 noise를 제거하도록 모델을 학습하여, 완전한 noise 이미지에서 이미지를 생성할 수 있게 된다. 
<br/><br/><br/>

## Forward process

앞서 이야기했듯 Diffusion process는 두단계에 거쳐 진행되는데, 첫번째는 바로 Forward Process $q$ 이다. 데이터 분포에서 샘플링된 데이터 포인트 $x_0$가 주어지면 여기에 <mark style='background-color: var(--hl-yellow)'><span style='color: var(--text-color)'>$T$ step에 나누어 소량의 Gaussian noise(fixed)를 더해가는 과정</span></mark>이다. 더해지는 Gaussian noise의 크기는 사전에 정의된다($\beta_t$). Foward Process의 마지막 스텝이 끝나면, 데이터 포인트는 완전한 noise image가 된다. 

![Diffusion_2](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/54aa5d97-1a96-46dc-9117-6f759b11b623){: width="700px"}
<br/><br/><br/>

## Reverse process

위의 과정을 반대로 진행할 수 있다면 Gaussian Noise에서 실제 sample을 다시 생성할 수 있다. Reverse process $p$는 noise image  $x_T$로 부터 원본 이미지 $x_0$를 복원하는 과정이다. 여러 스텝에 나누어 점진적으로 noise를 제거하고, 이 과정에서 noise를 제거한 각 step의 이미지는 forward process의 이미지와 같아야한다는 방식으로 학습을 진행한다. 여기서 Noise를 제거하는 <mark style='background-color: var(--hl-yellow)'><span style='color: var(--text-color)'>조건부확률을 근사하는 모델($p_{\theta}$)을 학습한다</span></mark>(뒤에서 추가적인 설명). 각 시점 별 평균과 분산을 모델이 예측하도록 해야한다.

![Diffusion_3](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/83808f2a-5d3b-40fc-930b-9cfd9cb3e855){: width="700px"}
<br/><br/>

정리하자면 Forward Process에서는 고정된 Gaussian noise를 단계별로 더하고, Reverse process에서는 학습된 모델에서 추정된 noise를 단계별로 제거하여 이미지를 생성한다.
<br/><br/><br/><br/><br/>


# Training Diffusion Model

---

Forward process에서 $q(x_t|x_{t-1})$을 수행하여 $x_t$를 만들었다면, 반대로 조건부 확률 $q(x_{t-1}|x_t)$을 알아내 $x_t$에서 $x_{t-1}$을 알아낼 수 있다. 이렇게 $x_t$에서 점진적으로 noise를 제거해가며 $x_0$를 만드는 것이 우리의 목표이다. 하지만 $q(x_{t-1}|x_t)$는 posterior probability(사후 확률)로, 매 시점에서의 prior probability $q(x_t)$를 알 수 없기 때문에 계산할 수 없다(Bayes rule). 결국 $q$를 잘 모델링할 수 있는 $p_{\theta}$를 학습해야한다. <br/><br/>
<mark style='background-color: var(--hl-green)'><span style='color: var(--text-color)'>확률 분포 $q$에서 관측한 값으로 확률분포 $p_{\theta}$의 likelihood를 구하였을 때, likelihood값이 최대가 되는 확률분포를 찾는 Maximum Likelihood Estimation 문제.</span></mark>
<br/><br/><br/>

그럼 마찬가지로 Maximum likelihood Estimation을 수행하는 VAE와 비교해보자. ([이전 글 참조: VAE 정리](https://lunaleee.github.io/posts/VAE/))

VAE와 Diffusion의 구조적인 차이를 비교해보면 아래 그림과 같다. VAE는 하나의 latent variable을 추출한다. 이와 달리 Diffusion은 **<mark style='background-color: var(--hl-yellow)'><span style='color: var(--text-color)'>Markov chain 형식</span></mark>**으로 여러개의 latent variable을 단계적으로 생성해낸다. 여기서 두 모델의 두드러지는 차이는 latent variable의 수가 다르다는 것으로 해석해 볼 수 있다. 이러한 차이는 Loss를 구성하는데도 차이를 나타내게 된다.

<details>
  <summary><b>Markov chain의 정의</b></summary>
  <div markdown="1">
  Markov chain, process는 Markov 특성(property)을 갖는 이산시간(discrete time) 확률과정(stochastic process)이다.
  각 시점에서의 상태가 주어졌을 때, 미래 시점 t+1에서의 상태는 과거 상태와 독립적으로 오로지 **<mark style='background-color: var(--hl-green)'><span style='color: var(--text-color)'>현재 상태 t에 의해서만 결정</span></mark>**된다는 것을 의미한다.
  </div>
</details>


![Diffusion_4](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/90d2ff6d-4182-43a5-a3ba-3fe4187ea6b8){: width="750px"}
<br/><br/><br/>

# Training Loss

---

## Diffusion Loss

그럼 이어서 Loss fuction 측면에서 VAE와 Diffusion을 비교하여 개념적으로 이해해보도록 하자.

![Diffusion_5](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/757520ff-07f9-4545-97ab-e5220e02ef68){: width="600px"}

먼저 VAE는 다음과 같은 Loss Function을 가지고 있다. VAE Loss fuction의 유도 과정은 역시 이전 글에서 정리했으니 참고해보자. VAE의 Loss는 Regularization term과 Reconstruction term으로 구분된다. 
<br/><br/><br/>

![Diffusion_6](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/f9d2b913-4617-4cf2-9b2c-9c265a90b151){: width="800px"}

Diffusion Loss 역시 비슷한 구조이다. VAE와 유사하게 Regularization term과 Reconstruction term을 가지고 있다. 

하지만 VAE에서는 없었언 **<mark style='background-color: var(--hl-yellow)'><span style='color: var(--text-color)'>Denoising Process</span></mark>** 관련 term이 추가되었다. Diffusion은 많은 수의 latent variable을 생성하는 Markov chain 형태를 가지고 있기 때문에, 이 부분이 Diffusion Process에서는 가장 중요한 부분이라고 볼 수 있다. $X_T$ 시점에서 $X_1$까지 이어지는 Denoising Process를 학습하도록 하는 Loss term이 이 부분이라고 볼 수 있기 때문이다.
<br/><br/>

Denoising Process term을 자세히 살펴보면, Forward process를 나타내는 conditional gaussian 분포$(q(-))$와, Reverse process를 나타내는 conditional gaussian 분포$(P_{\theta}(-))$간의 KL Divergence를 구하는 문제이다. 결국 **<mark style='background-color: var(--hl-yellow)'><span style='color: var(--text-color)'>Reverse process$(P_{\theta})$는 Forward process$(q)$를 최대한 approximation하도록 학습</span></mark>**되는 것이다.

![Diffusion_7](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/f0242d91-e5fe-4ed4-8012-a54d840b1479){: width="400px"}

VAE 식을 유도하는 과정에서 변형을 통해 위와 같은 식**($Loss_{Diffusion}$)**을 유도할 수 있는데, 해당 과정을 유도하는 과정은 생략하도록 하겠다. 이 과정은 [Blue collar Developer 블로그](https://developers-shack.tistory.com/8)에 자세히 유도되어 있으니 궁금하다면 참고해보면 좋을 것 같다.
<br/><br/><br/><br/>

## DDPM

![Diffusion_8](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/0366b4c3-93aa-4600-85f7-7a9864583160){: width="900px"}

위에까지 VAE에서 Diffusion Process에 관한 Loss를 도출했다. 이후 발표한 [5]DDPM 논문에서는 이러한 과정을 Network 관점에서 접근하기 위해 새로운 시각을 도입하여 Loss를 최적화했다. 해당 과정을 살펴보도록 하자.
<br/><br/><br/>

### 1. Regularization Term 제외

먼저 첫 번째 term인 Regularization term을 살펴보자. 해당 부분은 T시점의 latent variable이 특정한 prior 분포, 여기서는 Gaussian 분포를 따르도록 강제하는 역할을 하고 있다. 하지만 여기서 1000번의 step에 걸쳐 noise를 주입했을 때, <mark style='background-color: var(--hl-yellow)'><span style='color: var(--text-color)'>굳이 $\beta$를 학습하지 않아도 T시점의 latent variable이 isotropic gaussian과 매우 유사하게 형성</span></mark>된다고 한다.<br/><br/>
실제로 논문에서 $beta_1=10^{-1}$에서 $beta_T=0.02$까지 $T=1000$으로 linear하게 증가한다고 하며, 이에 따라 $q(x_T|x_0)$를 계산해보면 최종적인 latent variable$(z_T = x_T)$은 아래와 같은 결과를 얻을 수 있다. <br/><br/>

$$
q(x_T|x_0) = N(x_T; .00635x_0, 0.99995I) :≈﻿​ N(0,I)
$$

<br/>
이렇듯 거의 gaussian과 유사한 것을 확인할 수 있다. 따라서  $\beta$에 관한 부분은 학습하지 않고 해당 term은 constant로, 제외한다.
<br/><br/><br/>

### 2. Reconstruction Term 제외

다음은 세번째 term인 reconstruction term이다. 해당 부분은 VAE의 reconstruction loss와 같은 역할을 한다. 하지만 VAE는 decoder를 거쳐 최종 재생성한 이미지와 원래 이미지간의 reconstruction error였던데 비해 해당 부분은 <mark style='background-color: var(--hl-yellow)'><span style='color: var(--text-color)'>latent $x_1$으로부터 $x_0$을 추정하는 확률 모델을 최적화</span></mark>하고자한다. 즉 한 step의 아주 미세한 noise 이미지에서 원래 이미지를 복구하는 부분으로  모델입장에서는 보다 쉬운 task가 된다. 따라서 해당 term도 제외한다.
<br/><br/><br/>

### 3. Denoising Process 재구성

![Diffusion_9](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/38eb95dd-9f76-403e-a92c-acb0bdb3cfc0){: width="1000px"}

그럼 이제 가장 중요한 Denoising process term을 최적화해보자.
<br/><br/>

앞서 언급했듯, Denoising Process(Reverse Process)는 조건부 gaussian 분포의 **<mark style='background-color: var(--hl-yellow)'><span style='color: var(--text-color)'>각 시점별 평균과 분산을 예측</span></mark>**하는 문제이다.

![Diffusion_10](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/4cde21e3-e535-4a72-ae37-49c6abd24bfc){: width="600px"}

앞서 Regularization term을 제거했던 것과 같은 논리로, 주입된 noise의 크기 $\beta$는 미리 사전 정의한 알고있는 값이다. 따라서 알고있는 $\beta$를 활용하여 분산을 제외하기로 한다. **<mark style='background-color: var(--hl-green)'><span style='color: var(--text-color)'>학습 대상이었던 분산을 각 시점의 누적된 noise 크기로 상수화</span></mark>**한다.

이로서 학습 대상이 각 시점별 조건부 gaussian 분포의 평균을 추정하는 문제로 축소되었다.
<br/><br/>

![Diffusion_11](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/347310b0-966d-4795-8ce1-ae17bcdd5712){: width="900px"}

여기서 Denoising Process식으로 돌아와, **<mark style='background-color: var(--hl-green)'><span style='color: var(--text-color)'>Denoising Process를 mean function 추정 관점</span></mark>**에서 풀어보면 q와 p를 아래식과 같이 풀어낼 수 있다. 그리고 두 함수 사이의 KLD는 오른쪽 수식(1)과 같이 정의할 수 있다.
<br/><br/>

이 식에서 학습 대상인 mean function을 denoising 관점에서 새롭게 정의하기 위해 **<mark style='background-color: var(--hl-yellow)'><span style='color: var(--text-color)'>Reparameterization trick</span></mark>**을 적용하였다(DDPM의 contribution). 
<br/><br/>

이를 위해 x_t를 x_0에 대한 reparameterization 형태로 나타내고 식을 정리하면 아래식을 얻을 수 있다. <br/><br/>

$$
\mathbf{x}_t = \sqrt{\bar\alpha_t}\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\epsilon    \quad\quad\quad ; where \epsilon_{t-1}, \epsilon_{t-1}, ... \sim N(0, I)
$$

<br/>
<details>
  <summary>위 식에 대한 계산</summary>
  <div markdown="1">
  $$
  \quad\mathbf{x}_t = \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{1-\alpha_t}\epsilon_{t-1}     \quad\quad\quad\quad\quad      ; where\; \epsilon_{t-1}, \epsilon_{t-1}, ...\; \sim N(0, I)
  $$
  $$
  \quad\quad= \sqrt{\alpha_t\alpha_{t-1}}\mathbf{x}_{t-2} + \sqrt{1-\alpha_t\alpha_{t-1}}\bar\epsilon_{t-2}   \quad\quad  ; where\; \bar\epsilon_{t-2} merge\; two\; Gaussian(*).\quad\quad\quad
  $$
  $$
  \quad\quad = … \qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad
  $$
  $$
  \quad\quad= \sqrt{\bar\alpha_t}\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\epsilon \qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad
  $$
  $$
  \quad(\bar{\alpha}:= \prod^t_{s=1}\alpha_s, \; \alpha_t = 1-\beta_t)
  $$
  </div>
</details>

<br/>
위의 식에서 $\alpha$는 $\beta$를 통해 계산할 수 있는 노이즈의 크기를 나타내는 값이라 알고 있는 값이고, $x_0$ 또한 알고 있는 값이므로 해당 식에서 결국 모르는 값, 즉 학습해야할 것은 $\epsilon$이 된다.
<br/><br/>

![Diffusion_12](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/553ebdb9-3479-41d3-8541-70d4d3a799cb){: width="650px"}

이렇게 reparameterization으로 풀어낸 $x_0$를 위의 수식 (1)에 대입하면 수식 (2)와 같이 정리할 수 있다. 수식 (2)에 따르면, 학습해야하는 mean function은 주어진 시점에 파란색 밑줄 부분에 해당하는 식을 예측해야함을 알 수 있다. 해당 식에서 남은 예측 대상은 noise $\epsilon$ 뿐인 점을 고려하여, $\epsilon$에 $\theta$를 부여(학습 파라미터)하여 식을 정리하면 수식 (3)을 얻을 수 있다. 
<br/><br/>

![Diffusion_13](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/4a0d7797-2ac5-46f5-a221-9095ac30c223){: width="750px"}

여기서 수식 (2)와 수식 (3)을 조합하면 수식 (4)와 같은 새로운 목적식이 유도된다. 수식 (4)를 살펴보면, 결국 DDPM model($\epsilon_\theta$)이 학습해야 할 것은 주어진 t시점의 gaussian noise($\epsilon$) 가 된다. 이처럼 **<mark style='background-color: var(--hl-yellow)'><span style='color: var(--text-color)'>각 시점의 다양한 scale의 gaussian noise를 예측해, denoising에 활용</span></mark>**하고자 하는 것이 DDPM의 목적이라고 볼 수있다.
<br/><br/>

최종적으로는 coefficient term을 제거하여(문제를 단순화하기 위함) 수식 (5)와 같은 최종식을 얻을 수 있다. 

![Diffusion_14](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/d7095149-f923-4d72-bdc9-51a3d087e29b){: width="650px"}
<br/><br/><br/><br/>

# Summary

---

![Diffusion_15](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/36374faf-fe05-40aa-a3a1-7a578afbf711){: width="400px"}
_이미지 출처:https://arxiv.org/abs/2112.07804_

이렇게 대표적인 Generative model에 대해 정리해보았다. 각각의 모델이 장단점을 뚜렷하게 가지고 있는데, 생성모델의 3가지 특성을 통해 모델들을 분류한 그림은 위와 같다. 이를 Generative trilemma라고 한다.
<br/><br/>

마지막으로 앞서 정리해 본 Generative model 구조들을 간단하게 비교해보고 Generative model에 대한 정리를 마친다.

![Diffusion_1](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/dc9f48d6-2b91-4a33-9583-d2af4b494a8c){: width="1000px"}

![Diffusion_16](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/d014f527-c035-4fea-b360-d29c1f3a018f){: width="600px"}
<br/><br/><br/><br/><br/><br/>

---

**Reference**

[1] Lil’Log: [https://lilianweng.github.io/posts/2021-07-11-diffusion-models/](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)<br/>
[2] 고려대학교 산업경영공학 DSBA 연구실: [https://www.youtube.com/watch?v=_JQSMhqXw-4](https://www.youtube.com/watch?v=_JQSMhqXw-4)<br/>
[3] xoft: [https://xoft.tistory.com/32](https://xoft.tistory.com/32)<br/>
[4] DDPM(paper): [https://arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)
