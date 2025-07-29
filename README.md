# 잡음 제거
이 프로젝트는 transformer와 auto encoder를 사용해 28명의 
음성이 섞인 음원에서 하나의 음성만을 추출하는 AI 모델이다.
## 데이터셋
데이터셋으로는 valentini noisy speech을 사용했다. 그 중에서 
28명의 음성이 섞여있는 데이터셋을 사용. 

## 데이터 전처리
모델에 음원을 파형 형태로 입력하는것과 Spectogram
형태로 입력하는것 중 후자를 택했다. Spectogram을 입력으로
넣는것에는 몇가지 장점이 있다. 
#### 1. 주파수 성분을 학습가능
#### 2. 음원이 시각적으로 보여 CNN 가능

## 모델
Denoiser의 모델로 Autoencoder와 Transformer Encoder