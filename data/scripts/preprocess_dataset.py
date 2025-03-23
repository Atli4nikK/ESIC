import shutil
import random
import logging
from pathlib import Path
from typing import List

# Настроим логирование
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Пути
BASE_DATASET = Path("./data/dataset_v1")
NEW_DATASET = Path("./data/s3_downloaded/dataset")
S3_FOLDER = Path("./data/s3_downloaded")

TRAIN_DIR = BASE_DATASET / "train"
TEST_DIR = BASE_DATASET / "test"

# Доля данных для train и test
TRAIN_RATIO = 0.8

# Убеждаемся, что нужные папки существуют
TRAIN_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)

def is_folder_empty(path: Path) -> bool:
    """Проверяет, пустая ли папка."""
    return not any(path.iterdir())

def distribute_new_data(train_ratio: float = TRAIN_RATIO):
    """Распределяет новые данные по train и test."""
    if not NEW_DATASET.exists():
        logger.warning(f"Папка {NEW_DATASET} не найдена. Нечего распределять.")
        return

    for category_path in NEW_DATASET.iterdir():
        if not category_path.is_dir():
            continue

        logger.info(f"Обрабатываем класс: {category_path.name}")

        # Создаём папки в train и test
        train_category_path = TRAIN_DIR / category_path.name
        test_category_path = TEST_DIR / category_path.name
        train_category_path.mkdir(parents=True, exist_ok=True)
        test_category_path.mkdir(parents=True, exist_ok=True)

        # Получаем список всех файлов в категории
        new_files: List[Path] = [f for f in category_path.iterdir() if f.is_file()]
        random.shuffle(new_files)  # Перемешиваем файлы

        # Разделяем файлы на train и test
        split_index = int(len(new_files) * train_ratio)
        train_files, test_files = new_files[:split_index], new_files[split_index:]

        # Перемещаем файлы
        for file in train_files:
            file.rename(train_category_path / file.name)

        for file in test_files:
            file.rename(test_category_path / file.name)

        logger.info(f"Добавлено {len(train_files)} в train, {len(test_files)} в test")

    logger.info("✅ Добавление новых данных завершено!")

    # Проверяем, пустые ли все папки в s3_downloaded/dataset
    if all(is_folder_empty(folder) for folder in NEW_DATASET.iterdir()):
        logger.info("Все папки пустые, удаляем s3_downloaded...")
        shutil.rmtree(S3_FOLDER)
        logger.info("✅ s3_downloaded успешно удалён!")

if __name__ == "__main__":
    distribute_new_data()
