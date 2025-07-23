import os

import torch.utils.data
import torchaudio
import torch.nn.functional as F
import random

from torch.utils.data import Dataset

# 데이터셋 디렉토리
clean_path = "datasets/clean_trainset_28spk_wav"
noisy_path = "datasets/noisy_trainset_28spk_wav"


# 데이터셋 구성
class NoisySpeech(Dataset):
    def __init__(self, noisy_dir: str, clean_dir: str):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        # 디렉토리 내 파일 정렬
        self.noisy_files = sorted(os.listdir(self.noisy_dir))
        self.clean_files = sorted(os.listdir(self.clean_dir))

    def __getitem__(self, idx):
        noisy_file = os.path.join(self.noisy_dir, self.noisy_files[idx])
        clean_file = os.path.join(self.clean_dir, self.clean_files[idx])
        noisy_wave, _ = torchaudio.load(noisy_file)
        clean_wave, _ = torchaudio.load(clean_file)
        return noisy_wave, clean_wave

    def __len__(self):
        return len(self.noisy_files)


# 음성 샘플 길이를 모두 6초로 하기 위해 padding, truncate 함수 추가
def my_collate_fn(batch):
    new_noisy = []
    new_clean = []

    target_length = 288000

    for noisy, clean in batch:
        curr_length = noisy.shape[1]

        if curr_length < target_length:  # 6초보다 짧을 경우 sample rate 48000Hz * 6초 = 288,000 samples 로 패딩
            pad_length = target_length - curr_length
            noisy = F.pad(noisy, (0, pad_length))
            clean = F.pad(clean, (0, pad_length))

        elif curr_length > target_length:  # 6초보다 길 경우 임의의 구간에서부터 6초의 음원으로 truncate
            rand_start = random.randint(0, curr_length - target_length)  # 임의의 시작 시간 선정
            noisy = noisy[:, rand_start:rand_start + target_length]
            clean = clean[:, rand_start:rand_start + target_length]

        else:
            pass

        new_noisy.append(noisy)
        new_clean.append(clean)
    return torch.stack(new_noisy), torch.stack(new_clean)


train_dataset = NoisySpeech(noisy_path, clean_path)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=my_collate_fn)
