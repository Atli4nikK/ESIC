import os
import sys
import time
import warnings
import argparse
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from data.dataset import ESICDataModule
from models.baseline_model import MobileNetClassifier

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


torch.set_float32_matmul_precision("medium")

warnings.filterwarnings('ignore')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="./data/dataset_v1", type=str, help='path to dataset')
    parser.add_argument('--epochs', default=1, type=int, help='num of total training epochs')
    parser.add_argument('--batch-size', default=16, type=int, help='total batch size')
    parser.add_argument("--imgsz", type=int, default=224, help="train, val image size (pixels)")
    args = parser.parse_args()



    dm = ESICDataModule(dataset_path=args.data, imgsz=args.imgsz)

    model = MobileNetClassifier(num_classes=3)

    trainer = pl.Trainer(accelerator='gpu',
                         fast_dev_run=False,
                         max_epochs=args.epochs,
                         default_root_dir="./train/",
                         callbacks=[ModelCheckpoint(monitor='val_accuracy',
                                                    mode='max',
                                                    save_top_k=1,
                                                    filename='{epoch:02d}-{val_accuracy:.4f}', )],)


    start_time = time.monotonic()
    trainer.fit(model=model,
                datamodule=dm, )
    print(f"Training time {(time.monotonic() - start_time) / 60:.2f} min")

    trainer.validate(model=model,
                    datamodule=dm, )

    trainer.test(model=model,
                datamodule=dm,
                ckpt_path="best")

    # После завершения тренировки
    best_model_path = trainer.checkpoint_callback.best_model_path
    with open("./train/best_model.yaml", "w") as f:
        yaml.dump({"best_model_path": best_model_path}, f)



if __name__ == '__main__':
    main()
