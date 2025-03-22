import os

import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import transforms
import torch


# создаем класс датамодуль используя лайтнинг
class ESICDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path: str, transform: bool=True, batch_size: int=1, imgsz: int=224):
        super().__init__()
        self.dataset_path = dataset_path
        self.transform = transform
        self.batch_size = batch_size
        self.imgsz = imgsz
        # трансформации для изображений, ресайзим,
        # переводим в тензоры и нормализуем
        if transform:
            self.tfms = transforms.Compose([
                transforms.Resize((self.imgsz,self.imgsz)),
                transforms.ToTensor(),
                transforms.Normalize(std=[0.3481, 0.3755, 0.3738],
                                     mean=[0.6722, 0.6039, 0.6150]),
            ])
        else:
            self.tfms = transforms.ToTensor()

    def prepare_data(self):
        pass

    # метод для инициализцаии датасетов
    def setup(self, stage: str):
        # для метода фит
        if stage == "fit":
            self.train_ds = torchvision.datasets.ImageFolder(root=os.path.join(self.dataset_path,"train"),
                                                             transform=self.tfms)
            self.val_ds = torchvision.datasets.ImageFolder(root=os.path.join(self.dataset_path,"test"),
                                                           transform=self.tfms)
        # для метода тест
        if stage == "test":
            self.test_ds = torchvision.datasets.ImageFolder(root=os.path.join(self.dataset_path,"test"),
                                                            transform=self.tfms)
        # для метода предикт
        if stage == "predict":
            self.pred_ds = torchvision.datasets.ImageFolder(root=os.path.join(self.dataset_path,"test"),
                                                            transform=self.tfms)

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=self.batch_size,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                          batch_size=self.batch_size,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds,
                          batch_size=self.batch_size,
                          shuffle=False)

def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std

def main() -> None:
    # проверка работы датасета
    dm = ESICDataModule(dataset_path="./data/dataset_v1",
                        transform=True,
                        batch_size=16, )

    dm.setup(stage="fit")
    print(f"Train dataset: {len(dm.train_ds)} samples")
    print(f"Validation dataset: {len(dm.val_ds)} samples")

    train_loader = dm.train_dataloader()
    train_mean, train_std = get_mean_and_std(train_loader)
    print(train_mean, train_std)

    dm.setup(stage="test")
    print(f"Test dataset: {len(dm.test_ds)} samples")

    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx + 1}")
        print(f"Images shape: {images.shape}")
        print(f"Labels: {labels}")
        break

    train_loader = dm.train_dataloader()
    images, labels = next(iter(train_loader))  # Берём первый батч

    # Функция для денормализации изображения
    def denormalize(img):
        mean = np.array([0.6722, 0.6039, 0.6150])
        std = np.array([0.3481, 0.3755, 0.3738])
        img = img.numpy().transpose((1, 2, 0))  # Меняем оси: CxHxW -> HxWxC
        img = std * img + mean  # Обратная нормализация
        return np.clip(img, 0, 1)  # Ограничиваем значения от 0 до 1

    # Визуализация батча
    fig, axs = plt.subplots(4, 4, figsize=(10, 10))  # 4x4 сетка для 16 изображений
    axs = axs.flatten()  # Преобразуем в одномерный массив для удобства итерации

    for img, label, ax in zip(images, labels, axs):
        ax.imshow(denormalize(img))  # Денормализуем и показываем изображение
        ax.set_title(f"Label: {label.item()}")  # Отображаем метку
        ax.axis('off')  # Убираем оси

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()