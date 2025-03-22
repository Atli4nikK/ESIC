import os
import sys
import torch
import onnx
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from models.baseline_model import MobileNetClassifier

def convert():
    # Загружаем PyTorch-модель
    model = MobileNetClassifier(num_classes=3)
    model.load_state_dict(torch.load("./models/model.pth", map_location="cpu"))
    model.eval()

    # Подготовим тестовый вход (пример изображения)
    dummy_input = torch.randn(1, 3, 224, 224)  # Зависит от входного размера твоей модели

    # Экспортируем в ONNX
    torch.onnx.export(
        model, dummy_input, "./models/model.onnx",
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}  # Поддержка разного batch_size
    )

    print("✅ Модель успешно экспортирована в ONNX!")

def check():
    onnx_model = onnx.load("./models/model.onnx")
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX модель корректна!")

if __name__ == '__main__':
    convert()
    check()