---
title: "딥러닝 Normalization (Batch Normalization, Layer Normalization, Instance Normalization, Group Normalization 비교)"
author: lunalee
date: 2024-04-09 19:20:47 +0800
categories: [AI, Concept Note]
tags: [Basic, Normalization]
pin: false
math: true
---

<br/>

이번 게시물에서는 딥러닝에서 중요한 개념인 Normalization에 대해 정리해보자. Network를 구성할 때 layer의 앞 또는 뒤에 추가되는 normalization의 필요성에 대해 정리하고, 다양한 normalization 방법에 대해 차이점 위주로 간단하게 정리해 보자.
<br/><br/><br/><br/>

# Normalization 이란?

---

Normalization이란 **<mark style='background-color: var(--hl-yellow)'><span style='color: var(--text-color)'>데이터의 값을 특정 범위로 scaling 하는 것</span></mark>**을 의미한다. 입력 데이터를 딥러닝 모델에 넣기 전에, 전처리로 normalization을 수행하는 것은 일반적인 방법이다. 이렇게 입력 데이터를 특정 범위로 scaling하는 이유는 데이터 feature에 따라 범위가 다르기 때문이다. 
<br/><br/>

Network의 출력은 각 feature vector의 linear combination이다. 아래의 그림처럼 각각 다른 scale을 가지는 두개의 feature에 대해 network를 학습시킨다는 것은 network가 **서로 다른 scale을 가지는 feature에 대한 weight를 학습**한다는 것을 의미한다. 비교적 큰 scale을 가지는 feature에 작은 scale을 가지는 feature가 묻히는 현상이 발생할 수 있다. 

![Normalization_1.png](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/208f92cb-2903-4867-a043-67bc32725954){: width="550px"}
_이미지 출처: https://towardsdatascience.com/batch-norm-explained-visually-how-it-works-and-why-neural-networks-need-it-b18919692739_
<br/><br/>

아래의 그림은 **<mark style='background-color: var(--hl-green)'><span style='color: var(--text-color)'>gradient descent의 관점</span></mark>**에서 noramlization을 수행하지 않았을 경우 발생하는 문제점이다. 아래와 같이 두 가지 상황에 대해 살펴보자. 첫 번째 그림과 같은 경우에는 두 feature 중 $\text{W}_2$의 scale이 $\text{W}_2$의 scale 보다 확연히 넓은 범위를 가지고 있는 것을 볼 수 있다. 이러한 경우는 SGD 관점에서 확연히 **$\text{W}_2$ 축 방향 위주로 gradient를 계산**하게 된다. 이렇게 한 차원을 중심으로 이동하게 되므로 빠르게 수렴하지 못하고 더 많은 단계를 거쳐야한다.

![Normalization_2.png](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/ff1aba20-b265-4375-8065-3d9850286d83){: width="800px"}

두 feature가 동일한 scale을 가지는 두 번째 그림의 경우 **loss landscape는 더 균일**해지고, 비교적 **<mark style='background-color: var(--hl-yellow)'><span style='color: var(--text-color)'>완만하게 minimum으로 도달</span></mark>**할 수 있다.
<br/><br/><br/>

Normalization의 필요성에 대해 이해했으므로, 다음으로 입력데이터가 아니라 **hidden layer에서 feature를 특정 범위로 scaling**하는 Batch Normalization의 필요성에 대해 알아보자.
<br/><br/><br/><br/><br/><br/><br/>


# (Batch) Normalization의 필요성

---

앞서 언급했듯 입력 데이터 normalization은 network의 수렴을 빠르게 진행하는 장점이 있다.<br/>
그렇다면 hidden layer의 feature에 대해서는 왜 normalizatioin이 필요할까?
<br/><br/>

앞서 네트워크 입력 데이터에 대해서는 일정한 scale을 갖도록 조정해 주었다. 하지만 Network 내부에서 hidden layer를 거쳐 나오는 feature는 layer마다 또 다른 분포 특성을 갖게 된다. 이러한 현상을 **<mark style='background-color: var(--hl-green)'><span style='color: var(--text-color)'>Internal Covariate Shift</span></mark>**라고 부른다.
<br/><br/><br/>

## Internal Covariate Shift

Internal Covariate Shift 현상은 아래 그림과 같이 **<mark style='background-color: var(--hl-yellow)'><span style='color: var(--text-color)'>layer 별로 입력되는 값의 분포가 달라지는 현상</span></mark>**을 의미한다. Batch 단위로 학습을 진행하면 각각의 batch에 대해서 데이터 분포의 차이가 발생할 수 있다. 이렇게 batch마다 데이터 분포가 달라지면 각 layer에서는 다양한 분포에 대해 학습해야하고, 학습의 복잡성을 증가시킬 수 있다.

![Normalization_3.png](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/15a9946c-6ab3-4231-aa06-e4ab9920549f){: width="900px"}
<br/><br/>

즉, **입력 데이터에 적용한 원리와 동일한 방법이 hidden layer에서도 적용**됨으로서 **학습의 안정성이 증가하고 SGD가 더 잘 수렴**될 수 있다. 따라서 Batch norm과 같은 방법을 통해 입력 데이터에 사용한 것과 마찬가지로 hidden layer에서도 normalization을 진행하게 된다. 
<br/><br/>

방법은 동일하게 평균과 분산을 이용하여 진행하지만, 여기서 **<mark style='background-color: var(--hl-green)'><span style='color: var(--text-color)'>학습 가능한 affine parameter $\beta$와 $\gamma$</span></mark>**가 추가된다는 차이점이 있다. Normalization 공식은 다음과 같다.
<br/>

$$
\text{BN}(x_i) = \gamma \Big( \; \frac{X - \mu(x_i)}{\sigma(x_i)} \; \Big) + \beta
$$

<br/>

다음으로는 다양한 normalization 방법에 대해 알아보자. 이 방법들은 기본적으로 hidden layer 입력(feature)에 대해 평균, 분산을 사용하여 normalization하는 것은 동일하다. 단, 평균과 분산을 어떤 단위로 구할건지가 차이점이라고 볼 수 있다. 각각의 방법에 대한 차이점 위주로 알아보자.

<br/><br/><br/><br/><br/><br/>

# Batch Normalization

---

[Batch Normalization📄](https://arxiv.org/abs/1502.03167)은 2015년 ICML에 발표되었다.
<br/><br/>

Gradient를 구하여 network를 업데이트 할 때, 전체 학습 데이터에 대해 한번에 gradient를 구하고 평균/분산을 구하는 것은 불가능하기 때문에 일반적으로 학습시에 batch 단위로 업데이트를 진행하게 된다(stochastic gradient descent). 
<br/><br/>

마찬가지로 normalization을 위한 평균과 분산은 전체 데이터 집합에 대해 계산되어야하지만, 이것은 불가능하므로 Batch normalization에서는 전체를 대표하는 집합으로 Batch가 사용된다(따라서 이를 **mini-batch statics**라고 한다).<br/>
**<mark style='background-color: var(--hl-yellow)'><span style='color: var(--text-color)'>즉, Batch가 전체를 대표하는 집합이 되고, 각각의 Batch 별로 평균과 분포를 계산한다.</span></mark>** 
<br/>

$$
\mu_C(x) = \frac{1}{NHW} \sum^N_{n=1} \sum^H_{h=1} \sum^W_{w=1} x_{nchw}
$$

$$
\sigma_C(x) = \sqrt{\frac{1}{NHW} \sum^N_{n=1} \sum^H_{h=1} \sum^W_{w=1} (x_{nchw} - \mu(x))^2 \;+\; \epsilon }
$$

<br/>

![Normalization_4.png](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/7606f661-acac-4d98-a89b-b1c7370c13a5){: width="1000px"}<br/>
직관적으로 이해하면 위 그림과 같다.

Batch가 전체를 대표하는 집합이되므로 이론적으로는 Batch size가 클수록 좋다. 하지만 SGD 관점에서, local minima에 빠지거나 수렴속도가 느려지는 문제가 발생할 수 있으므로 실제 사용에서는 Batch size가 클수록 좋은 것은 아니다.
<br/><br/><br/><br/><br/>

## Batch Normalization의 Inference

### Training

각 mini-batch에서 mean, std를 구해 정규화를 진행하면 된다.  → **<mark style='background-color: var(--hl-green)'><span style='color: var(--text-color)'>mini-batch statics</span></mark>**

- parameter
    - $\gamma, \beta$:  $\gamma=1, \beta=0$으로 초기화되고, backprop을 통해 점차 학습된다.
    - $\mu, \sigma$:  mini-batch의 mean, std(학습되는 값 아님. 각각의 batch에서 계산).
<br/><br/>

### Inference

Training에서는 각 mini-batch의 평균, 분산을 구하면 되지만 inference시에는 batch 단위로 계산하지 않으므로, 평균과 분산을 구할 수 없다. 따라서 training 시에 각 mini-batch에서의 평균과 분산을 사용하고 버리는 것이 아니라, **<mark style='background-color: var(--hl-yellow)'><span style='color: var(--text-color)'>moving average를 사용하여 이 값을 축적</span></mark>**한다(매 mini-batch마다 업데이트된다). 각각의 batch에서 구한 값이 전체 데이터에 대해 계산되므로, inference시에는 mini-batch statics가 아닌 **<mark style='background-color: var(--hl-green)'><span style='color: var(--text-color)'>pouplar statics</span></mark>**를 사용한다고 볼 수 있다.

- parameter
    - $\gamma, \beta$:  training 시에 backprop을 통해 학습된 값.
    - $\mu, \sigma$:  training 시에 moving average로 계산된 mean, std
<br/><br/><br/><br/><br/>

## 문제점

- Batch Normalization은 training 중에 **mini-batch statics**를 사용하고 inference 중에 popular statics로 대체하여 계산하므로 **<mark style='background-color: var(--hl-yellow)'><span style='color: var(--text-color)'>training - inference 사이에 불일치</span></mark>**가 발생한다.(이를 해결하기 위한 Batch Renormalization등의 방법이 있음)
- Batch size를 키우기 어려운 RNN과 같은 계열에서는 좋은 성능을 기대하기 어렵다.
<br/><br/><br/><br/><br/><br/>

# Layer Normalization

---

다음은 2016년 발표된 [Layer Normalization📄](https://arxiv.org/abs/1607.06450)이다. 기본적으로 Normalizaiton을 진행하는 공식은 동일하므로 앞의 방법과 비교하여 평균과 분산을 구하는 집합에 대한 차이점 위주로 간단하게 이해해보자.
<br/><br/>

Layer Normalization은 평균과 분산을 batch 차원이 아닌, feature 차원으로 계산한다. BN이 채널별 평균, 분산($\mu, \sigma$)라고 한다면, LN은 데이터별 평균, 분산($\mu, \sigma$)이라고 볼 수 있다. 수식으로는 아래와 같다.
<br/>

$$
\mu_N(x) = \frac{1}{CHW} \sum^C_{c=1} \sum^H_{h=1} \sum^W_{w=1} x_{nchw}
$$

$$
\sigma_N(x) = \sqrt{\frac{1}{CHW} \sum^C_{c=1} \sum^H_{h=1} \sum^W_{w=1} (x_{nchw} - \mu(x))^2 \;+\; \epsilon }
$$

<br/>

![Normalization_5.png](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/f363153b-4028-4700-b4b3-6a56cae2cd37){: width="1000px"}<br/>
마찬가지로, 직관적으로 이해하면 위 그림과 같다.

LN은 batch size와 관계 없이 동작하므로, batch size를 키우기 어려운 **<mark style='background-color: var(--hl-yellow)'><span style='color: var(--text-color)'>RNN</span></mark>**과 같은 구조에서 좋은 성능을 보인다.
<br/><br/><br/><br/><br/><br/>

# Instance Normalization

---

마찬가지로 2016년 발표된 [Instance Noramlization📄](https://arxiv.org/abs/1607.08022)은 Layer Normalization과 비슷하지만, 각각 **<mark style='background-color: var(--hl-yellow)'><span style='color: var(--text-color)'>filter(Channel)에 관계없이 따로 정규화</span></mark>**된다. 즉, feature의 각 channel에 HW에 대해서만 정규화가 진행되므로, **<mark style='background-color: var(--hl-green)'><span style='color: var(--text-color)'>Contrast Normalization</span></mark>**의 효과를 얻을 수 있다고 보면 된다. 수식적으로는 아래와 같다.
<br/>

$$
\mu_I(x_c) = \frac{1}{HW}  \sum^H_{h=1} \sum^W_{w=1} x_{nchw}
$$

$$
\sigma_I(x_c) = \sqrt{\frac{1}{HW} \sum^H_{h=1} \sum^W_{w=1} (x_{nchw} - \mu(x_c))^2 \;+\; \epsilon }
$$

<br/>

![Normalization_6.png](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/eb09e117-1a93-4250-a501-fa7e45c10788){: width="1000px"}
<br/>

이러한 효과 때문에 **Style Transfer**와 같은 task를 수행할 때 주로 사용된다. Style transfer에서는 아래 그림과 같이 Content image를 Style image의 텍스처나 화풍으로 변환하는 것을 의미하는데, 이 때 **content image는 구조적인 부분이 중요하고 contrast에 의존하지 않아야**한다. 따라서 이러한 Task에 IN이 효과적으로 동작한다고 볼 수 있다.

![Normalization_7.png](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/ccf70ab0-4157-46f0-bc51-e50b30064010){: width="500px"}

마찬가지로, **<mark style='background-color: var(--hl-yellow)'><span style='color: var(--text-color)'>Class label이 입력 이미지의 명암(intensity)에 의존하지 않아야 하는</span></mark>** 이미지 분류에도 IN이 사용된다.(e.g.밤이나 낮에 촬영한 이미지라고 해도 강아지는 여전히 강아지인 것과 같은 문제)
<br/><br/><br/><br/><br/><br/>

# Group Normalization

---

앞서 BN은 feature map의 mean, std를 batch 단위로 계산해서 정규화를 진행한다. 이 때, 계산되는 mean, std는 batch 안의 데이터만 이용해서 계산되지만, batch가 충분히 크다면 이 mean, std가 데이터셋 전체의 mean, std를 대표할 수 있다는 가정을 전제로 한다.

하지만 BN은 batch의 크기가 작으면 이 가정을 만족시키기 어렵고, 구해지는 mean, std도 매 iteration(각 mini-batch)마다 달라지게 된다. 이러한 문제를 해결하기 위해 나온 방법이 Group Normalization이다.
<br/><br/><br/>

2018년 ECCV에 [Group Normalization📄](https://arxiv.org/abs/1803.08494) 논문이 발표되었으며, 형태적으로 IN과 LN이 절충된 형태라고 볼 수 있다. IN과 LN은 Batch size에는 독립적으로 동작하지만, Visual Recognition 분야에서는 좋은 성능을 보장하지 않았다. 

Group Normalization에서는 **<mark style='background-color: var(--hl-yellow)'><span style='color: var(--text-color)'>각 채널을 N개의 “그룹” 단위로 나눠서 정규화를 진행</span></mark>**한다. 채널당 그룹의 수 G는 hyperparameter이다. 여기서 G=1(그룹이 1개)이면 IN과 동일, G=C(그룹수가 채널수와 동일)인 경우 LN과 동일해진다. 평균과 분산에 대한 식은 아래와 같다.
<br/>

$$
\mu_G(x) = \frac{1}{GHW} \sum^G_{g=1} \sum^H_{h=1} \sum^W_{w=1} x_{nghw}
$$

$$
\sigma_G(x) = \sqrt{\frac{1}{GHW} \sum^G_{g=1} \sum^H_{h=1} \sum^W_{w=1} (x_{nghw} - \mu(x))^2 \;+\; \epsilon }
$$

<br/>

![Normalization_8.png](https://github.com/cotes2020/jekyll-theme-chirpy/assets/34572874/0bd55c9d-f083-4b07-a2fd-51303a44c926){: width="1000px"}<br/>
직관적으로 이해하면 위 그림과 같다.

만약 channel=6이고, group=2이면 한 그룹당 채널 수는 3개인 것이다. Group Normalization은 Visual Recognition task에서 Batch size=32에 대해 BN과 비슷한 성능을 보이고, **<mark style='background-color: var(--hl-yellow)'><span style='color: var(--text-color)'>Batch size가 32이하인 경우에 대해서는 BN보다 좋은 성능</span></mark>**을 보인다고 한다.
<br/><br/><br/><br/><br/><br/>


---

#### Reference

[1] [https://www.jeremyjordan.me/batch-normalization/](https://www.jeremyjordan.me/batch-normalization/)<br/>
[2] [https://towardsdatascience.com/batch-norm-explained-visually-how-it-works-and-why-neural-networks-need-it-b18919692739](https://towardsdatascience.com/batch-norm-explained-visually-how-it-works-and-why-neural-networks-need-it-b18919692739)<br/>
[3] [https://gaussian37.github.io/dl-concept-batchnorm/](https://gaussian37.github.io/dl-concept-batchnorm/)
