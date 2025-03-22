import timm
import torch
from torch import nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
import pytorch_lightning as pl


class MobileNetClassifier(pl.LightningModule):
    def __init__(self, num_classes: int, learning_rate: float = 1e-3, pretrain: bool = True, transfer:bool = False):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.pretrain = pretrain
        self.transfer = transfer

        self.example_input_array = torch.randn(1, 3, 224, 224)

        # Создаем MobileNet из timm
        self.model = timm.create_model('mobilenetv3_large_100', pretrained=self.pretrain, num_classes=self.num_classes)

        if self.transfer:
            # Заморозим все слои
            for param in self.model.features.parameters():
                param.requires_grad = False
            # Заменяем последний слой
            in_features = self.model.get_classifier().in_features
            self.model.fc = nn.Linear(in_features, self.num_classes)

        # Функция потерь
        self.criterion = nn.CrossEntropyLoss()

        # Метрики
        self.accuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self.precision = MulticlassPrecision(num_classes=self.num_classes, average='macro')
        self.recall = MulticlassRecall(num_classes=self.num_classes, average='macro')
        self.f1_score = MulticlassF1Score(num_classes=self.num_classes, average='macro')

    def forward(self, x):
        return self.model(x)

    def common_step(self, batch, stage):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        preds = outputs.argmax(dim=1)

        # Логирование метрик
        acc = self.accuracy(preds, labels)
        prec = self.precision(preds, labels)
        rec = self.recall(preds, labels)
        f1 = self.f1_score(preds, labels)

        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_accuracy", acc, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_precision", prec, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_recall", rec, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_f1_score", f1, prog_bar=True, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.common_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self.common_step(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        images, _ = batch
        outputs = self(images)
        preds = outputs.argmax(dim=1)
        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


def main() -> None:
    model = MobileNetClassifier(num_classes=2)
    # print(model)
    image = torch.rand(1, 3, 224, 224)
    out = model(image)
    print(out)


if __name__ == "__main__":
    main()