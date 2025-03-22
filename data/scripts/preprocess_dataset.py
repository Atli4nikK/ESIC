import os
import shutil
import random

# Пути
BASE_DATASET = "./data/dataset_v1"
NEW_DATASET = "./data/s3_downloaded/dataset"
S3_FOLDER = "./data/s3_downloaded"

TRAIN_DIR = os.path.join(BASE_DATASET, "train")
TEST_DIR = os.path.join(BASE_DATASET, "test")

# Доля данных для train и test
TRAIN_RATIO = 0.8

# Убеждаемся, что нужные папки существуют
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)


def is_folder_empty(path):
    """Проверяет, пустая ли папка"""
    return not any(os.listdir(path))


def distribute_new_data():
    for category in os.listdir(NEW_DATASET):
        new_category_path = os.path.join(NEW_DATASET, category)

        # Пропускаем, если это не папка
        if not os.path.isdir(new_category_path):
            continue

        print(f"Обрабатываем класс: {category}")

        # Создаём папки в train и test, если их нет
        train_category_path = os.path.join(TRAIN_DIR, category)
        test_category_path = os.path.join(TEST_DIR, category)
        os.makedirs(train_category_path, exist_ok=True)
        os.makedirs(test_category_path, exist_ok=True)

        # Получаем список всех новых файлов
        new_files = [f for f in os.listdir(new_category_path) if os.path.isfile(os.path.join(new_category_path, f))]
        random.shuffle(new_files)  # Перемешиваем файлы

        # Разделяем файлы на train и test
        split_index = int(len(new_files) * TRAIN_RATIO)
        train_files = new_files[:split_index]
        test_files = new_files[split_index:]

        # Копируем файлы
        for file in train_files:
            shutil.move(os.path.join(new_category_path, file), os.path.join(train_category_path, file))

        for file in test_files:
            shutil.move(os.path.join(new_category_path, file), os.path.join(test_category_path, file))

        print(f"Добавлено {len(train_files)} в train, {len(test_files)} в test")

    print("Добавление новых данных завершено!")

    # Проверяем, пустые ли все папки в s3_downloaded/dataset
    all_empty = all(is_folder_empty(os.path.join(NEW_DATASET, folder)) for folder in os.listdir(NEW_DATASET))

    if all_empty:
        print("Все папки пустые, удаляем s3_downloaded...")
        shutil.rmtree(S3_FOLDER)
        print("s3_downloaded успешно удалён!")


if __name__ == "__main__":
    distribute_new_data()
