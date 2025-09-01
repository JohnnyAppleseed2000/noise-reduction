import os, random
import torch
import torchaudio
import torch.nn.functional as F
import IPython.display as ipd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) 모델 불러오기
model = Denoiser().to(device)
checkpoint_path = '/content/drive/MyDrive/denoiser_checkpoint.pth'
ckpt = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# 2) 파일 고르기
root = "/root/.cache/kagglehub/datasets/muhmagdy/valentini-noisy/versions/3"
noisy_dir = os.path.join(root, "noisy_trainset_28spk_wav")
clean_dir = os.path.join(root, "clean_trainset_28spk_wav")

files = sorted(os.listdir(noisy_dir))
fname = random.choice(files)
noisy_path = os.path.join(noisy_dir, fname)
clean_path = os.path.join(clean_dir, fname)

noisy_wave, sr_n = torchaudio.load(noisy_path)  # (C, T)
clean_wave, sr_c = torchaudio.load(clean_path)
assert sr_n == sr_c, "샘플레이트가 다릅니다."
sr = sr_n

# 3) 6초(= 288000 @48kHz)로 청크 분할 + 마지막 패딩
TARGET_LEN = 6 * sr  # 288000


def split_pad(wav: torch.Tensor, chunk_len: int):
    T = wav.shape[1]
    chunks = []
    ptr = 0
    while ptr < T:
        end = min(ptr + chunk_len, T)
        chunk = wav[:, ptr:end]
        if chunk.shape[1] < chunk_len:
            chunk = F.pad(chunk, (0, chunk_len - chunk.shape[1]))
        chunks.append(chunk)
        ptr += chunk_len
    return chunks


noisy_chunks = split_pad(noisy_wave, TARGET_LEN)
clean_chunks = split_pad(clean_wave, TARGET_LEN)

# 4) 모델에 입력 → (복소) STFT를 iSTFT로 시간영역 복원
n_fft = 1024
hop = 256
win = 1024
window = torch.hann_window(win, device=device)

recon_chunks = []
with torch.no_grad():
    for n_chunk, c_chunk in zip(noisy_chunks, clean_chunks):
        n_chunk = n_chunk.to(device)  # (1, T)
        c_chunk = c_chunk.to(device)

        # 모델 추론 (아래는 예시 시그니처)
        recon_spec, noisy_mag, clean_mag = model(n_chunk, c_chunk)

        # iSTFT 복원 (채널 차원에 따라 squeeze/unsqueeze 조정)
        # recon_spec: (F, T) 혹은 (1, F, T)라고 가정
        recon_spec = recon_spec.squeeze(1)

        print(recon_spec.shape)
        recon_wave = torch.istft(recon_spec,
                                 n_fft=n_fft, hop_length=hop, win_length=win,
                                 window=window, length=TARGET_LEN, center=True)

        recon_chunks.append(recon_wave.cpu())

# 청크 합치기
recon_full = torch.cat(recon_chunks, dim=1)  # (1, total_T)
clean_full = torch.cat([c.cpu() for c in clean_chunks], dim=1)

# 5) 오디오로 듣기 (첫 6초만 미리보기)
ipd.display(ipd.Audio(recon_full.squeeze(0).numpy(), rate=sr))
ipd.display(ipd.Audio(clean_full.squeeze(0).numpy(), rate=sr))
