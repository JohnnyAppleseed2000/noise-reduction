import torch
import torch.nn as nn
import torch.optim as optim
from models import Denoiser
from data import dataset


# 1. GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. 모델 초기화, 데이터셋 불러오기
model = Denoiser().to(device)
train_loader = dataset.train_loader

# 3. 손실 함수 및 옵티마이저 정의
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 4. 학습 루프
num_epochs = 10
train_losses = []
for epoch in range(num_epochs):
    running_loss = 0.0
    for noisy_wave, clean_wave in train_loader:
        noisy_wave = noisy_wave.to(device)
        clean_wave = clean_wave.to(device)

        # Forward
        recon_noisy, clean_spec = model(noisy_wave, clean_wave)
        loss = criterion(clean_spec, recon_noisy)  # 입력과 출력 비교

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}")

print("훈련 완료!")

torch.save(model.state_dict(),'autoencoder.pth')