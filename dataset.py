from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import torchvision.utils as vutils



def load_data(img_size=64, batch_size=128) -> DataLoader:
    # 以 64x64 为例（适合较常见的 UNet 配置）
    transform = transforms.Compose([
        transforms.CenterCrop(178),  # 中心裁剪成方形
        transforms.Resize(img_size),  # Resize 到 64x64
        transforms.ToTensor(),  # 转为 [0, 1] 张量
        transforms.Normalize([0.5] * 3, [0.5] * 3)  # 标准 Normalize → [-1, 1]
    ])
    celeba_path = "./data/img_align_celeba"  # 包含 images/ 子文件夹
    dataset = ImageFolder(root=celeba_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    return dataloader

def show_batch(images):
    grid = vutils.make_grid(images[:64], nrow=8, normalize=True)
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.show()

if __name__ == '__main__':
    images, _ = next(iter(load_data()))
    print(f"图像个数: {images.shape[0]}")
    print(f"图像尺寸: {images.shape[2]} x {images.shape[3]}")
    show_batch(images)