import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from models import Denoiser
from data import NoisySpeech


# 1. GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. 모델 초기화, 데이터셋 불러오기
model = Denoiser().to(device)
train_dataset = NoisySpeech(noisy_dir, clean_dir)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=my_collate_fn, pin_memory=True)

# 3. 손실 함수 및 옵티마이저 정의
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# 4. 체크포인트 불러오기
checkpoint = 'False'
checkpoint_path = '/content/drive/MyDrive/denoiser_checkpoint.pth'
if os.path.exists(checkpoint_path) and checkpoint=='True':
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    train_losses = checkpoint['train_losses']
    print(f"Checkpoint 로드 완료 — {start_epoch} 에포크부터 재시작")
else:
    start_epoch = 0
    train_losses = []

# 5. 학습 루프
num_epochs = 30
for epoch in range(num_epochs):
    running_loss = 0.0
    with tqdm(train_loader, desc=f"Epoch {start_epoch+1}", dynamic_ncols=True) as pbar:
      for step, (noisy_wave, clean_wave) in enumerate(pbar):
          noisy_wave = noisy_wave.to(device)
          clean_wave = clean_wave.to(device)
          # Forward
          recon_noisy, noisy_mag, clean_mag = model(noisy_wave, clean_wave)
          loss = criterion(noisy_mag, clean_mag)  # 입력과 출력 비교
          # Backward
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          running_loss += loss.item()
        # tqdm 줄에만 업데이트되게 하기 (print 없이)
          pbar.set_postfix({
              "Step": f"{step+1}",
              "Loss": f"{loss.item():.8f}",
          })
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.8f}")
    scheduler.step(epoch_loss)

    # Checkpoint 저장
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses' : train_losses
    }
    torch.save(checkpoint, '/content/drive/MyDrive/denoiser_checkpoint.pth')

print("훈련 완료!")