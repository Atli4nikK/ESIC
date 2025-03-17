import os
import time
import shutil
import argparse

from PIL import Image
import timm
import torch
from torchvision import transforms
import requests
from torch.utils.data import DataLoader, Dataset


def load_imagenet_labels():
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError("Failed to download ImageNet labels")
    return response.json()

class ImageDataset(Dataset):
    def __init__(self, image_paths, preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image)
        return image_tensor, image_path

def is_car_present_batch(batch_images, model, labels, car_classes, threshold=0.5):
    with torch.no_grad():
        outputs = model(batch_images)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        batch_results = []
        for probs in probabilities:
            class_id = torch.argmax(probs).item()
            class_name = labels[class_id]
            if class_name in car_classes and probs[class_id] >= threshold:
                batch_results.append(True)  # Автомобиль найден
            else:
                batch_results.append(False)  # Автомобиля нет
        return batch_results

def main(input_folder, output_file, no_cars_folder, batch_size=16):
    start_time = time.time()

    # Если есть ведюха, инференсит на ней
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Загрузка предобученной модели
    print("Loading model...")
    model = timm.create_model('resnet50', pretrained=True)
    model.to(device)
    model.eval()

    # Загрузка меток ImageNet
    print("Loading ImageNet labels...")
    labels = load_imagenet_labels()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=model.default_cfg['mean'], std=model.default_cfg['std'])
    ])

    # Категории автомобилей для фильтрации
    car_classes = [
        "convertible", "sports car", "minivan", "pickup truck",
        "ambulance", "fire engine", "cab", "jeep", "limousine",
        "car", "truck", "bus", "automobile", "taxi", "estate car",
        "car wheel", "passenger car", "race car", "racing car",
        "sports car", "sport car", "fire truck", "police van",
        "police wagon"
    ]

    # Получение списка изображений
    image_paths = [
        os.path.join(input_folder, filename)
        for filename in os.listdir(input_folder)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    # Создание датасета и загрузчика данных
    dataset = ImageDataset(image_paths, preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Если папки для переноса нет, создаем ее
    os.makedirs(no_cars_folder, exist_ok=True)

    no_car_images = []

    print("Processing images...")
    for batch_images, batch_paths in dataloader:
        batch_images = batch_images.to(device)
        batch_results = is_car_present_batch(batch_images, model, labels, car_classes)

        for result, image_path in zip(batch_results, batch_paths):
            if not result:
                no_car_images.append(os.path.basename(image_path))
                shutil.move(image_path, os.path.join(no_cars_folder, os.path.basename(image_path)))

    # Сохраняем результат
    with open(output_file, 'w') as f:
        for filename in no_car_images:
            f.write(filename + '\n')

    elapsed_time = time.time() - start_time
    print(f"Processing completed in {elapsed_time:.2f} seconds.")
    print(f"Results saved to {output_file}")
    print(f"Images without cars moved to {no_cars_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect images without cars.")
    parser.add_argument("--input_folder", default="images", type=str, help="Path to the input folder with images.")
    parser.add_argument("--output_file", default="check.txt", type=str, help="Path to the output file.")
    parser.add_argument("--no_cars_folder", default="no_cars", type=str, help="Path to the folder for images without cars.")
    args = parser.parse_args()

    main(args.input_folder, args.output_file, args.no_cars_folder)
