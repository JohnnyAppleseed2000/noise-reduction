## 잡음 제거 [ Denoiser Model ] ##
이 잡음 제거 프로젝트는 딥러닝 기반의 오디오 잡음제거 모델입니다. 
모델 구조는 STFT → CNN → Transformer 구조를 사용하여 여러명의 발화자의
음성이 섞인 음원에서 한 사람만의 음성을 추출합니다.

[🔊 데모 페이지에서 바로 재생하기] https://johnnyappleseed2000.github.io/noise-reduction/



---
## 주요 특징
- Waveform 형태의 음원으로 **STFT 변환**을 통한 주파수 영역 정보를 모델에 입력
- **CNN**으로 지역 특성을 추출하고 차원축소
- **Transformer**를 통한 시간적 관계 학습
- **MSE Loss** 손실 함수 기반 학습
---
## 데이터셋
학습을 위해 사용된 데이터셋은 
**Valentini Noisy Speech Database**로, 28명의 발화자의 
음성이 섞인 noisy trainset 과 한 사람의 음성만이 있는 
clean trainset을 사용했습니다. 모든 음원의 Sample Rate는 48kHz입니다.
모델의 최적화를 위해 기존 데이터셋에 몇가지 전처리 방식을 
사용했습니다.

### 데이터 전처리
- 모든 음원을 6초로 통일했습니다. 6초보다 길면 임의의 지점에서 6초 길이의 음원 사용, 
6초보다 짧으면 6초에 맞게 padding 진행했습니다.
**(모든 음원을 6초로 통일함으로써 batch화가 단순해지고 긴 샘플이
입력되었을시 메모리 폭증을 예방할 수 있습니다)**
- Waveform 형태 그대로가 아닌 STFT 변환을 통해 spectogram 형태의
데이터를 입력했습니다. Spectogram을 통해 아래 그림과 같이 
시간축과 주파수축으로 나누어 음원 정보를 시각적으로 나타나게 합니다.
![librosa-stft-1.png](images%2Flibrosa-stft-1.png)
**Spectogram을 사용함으로써 CNN으로 하모닉과 같은 주파수 영역에 반복적으로
뜨는 패턴을 추출할 수 있고 Transformer를 통해 시간축 상호작용을
학습할 수 있습니다.**
- 나은 학습을 위해 Spectogram을 log scale로 변환 후 정규화 진행

---
## 모델
### 1. CNN Encoder
- 입력: log-magnitude STFT 스펙트로그램 
(Batch_size, 1 , 513 , 1126)
(B,1,513,1126) (48kHz·6초, n_fft=1024, hop=256).
- 역할: 하모닉/온셋 등 국소 시간–주파수 패턴 추출 + 다운샘플
- 1차 encoding: 1×513×1126→ 8×257×563
- 2차 encoding: 8×257×563→16×129×282
- 3차 encoding: 16×129×282→32×65×141

### 2. Transformer
입력 정리: 인코더 출력
(B,32,65,141) → (Time, Batch Size, Channel * Frequency)
=(141 , B , 2080) 로 reshape 후 Linear Projection 을 통해 차원 축소 
(2080→512).
### 3. CNN Decoder
ㅇㅇ


---
## 학습
- **손실 함수**: MSE Loss  
- **Optimizer**: Adam (lr=0.001, weight decay=1e-5)  
- **Batch Size**: 32  
- **Epochs**: 30
- **Early Stopping**: validation loss가 개선되지 않으면 조기 종료  
 

학습 과정에서 train loss는 점차 감소하였고, validation loss 기준으로 최적의 모델 가중치를 저장하여 과적합을 방지했습니다.

---
## 복원





