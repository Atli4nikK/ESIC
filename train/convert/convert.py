import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from models.baseline_model import MobileNetClassifier

# Загружаем чекпоинт
ckpt_path = "./models/model.ckpt"
checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))

# Создаём модель и загружаем веса
model = MobileNetClassifier.load_from_checkpoint(ckpt_path)
model.eval()  # Переводим в режим инференса

# Сохраняем только веса в формате .pth
torch.save(model.state_dict(), "./models/model.pth")
print("Модель сохранена в model.pth")
