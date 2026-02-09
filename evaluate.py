from torchmetrics.image.fid import FrechetInceptionDistance
import torch

from torch.utils.data import DataLoader
from torchvision import transforms

import os
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class CelebADataset(Dataset):
    def __init__(self, image_dir, partition_file, split='train', transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.filenames = []

        with open(partition_file, 'r') as f:
            for line in f:
                filename, split_label = line.strip().split()
                if (split == 'train' and split_label == '0') or \
                   (split == 'val'   and split_label == '1') or \
                   (split == 'test'  and split_label == '2'):
                    self.filenames.append(filename)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.filenames[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


transform = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

image_root = 'data/img_align_celeba/img_align_celeba'
partition_file = 'data/img_align_celeba/list_eval_partition.txt'

train_dataset = CelebADataset(image_root, partition_file, split='train', transform=transform)
test_dataset  = CelebADataset(image_root, partition_file, split='test',  transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
test_loader  = DataLoader(test_dataset,  batch_size=128, shuffle=False, num_workers=4)

fid = FrechetInceptionDistance(feature=2048, normalize=True).cuda()

# 添加真实图像（只做一次）
for batch in test_loader:
    fid.update(batch.cuda(), real=True)

# 添加生成图像（假设你有一个生成函数 sample_fn）
# for _ in range(len(test_loader)):
#     fake_images = sample_fn(batch_size=128).cuda()
#     fid.update(fake_images, real=False)

print("FID Score:", fid.compute().item())