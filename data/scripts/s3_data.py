import os
import logging
from pathlib import Path
from typing import Optional

import boto3
import yaml
from botocore.exceptions import BotoCoreError, NoCredentialsError


# Настроим логирование
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Пути и файлы
CONFIG_PATH = "s3_data_cfg.yaml"
LOCAL_ROOT = Path("./data/s3_downloaded")

# Функция загрузки конфигурации
def load_config(config_path: str) -> dict:
    """Загружает конфигурацию из YAML-файла."""
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Файл конфигурации {config_path} не найден.")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Ошибка парсинга YAML: {e}")
        raise

# Функция создания клиента S3
def create_s3_client(config: dict):
    """Создает и возвращает клиент S3."""
    try:
        return boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", config["aws_access_key_id"]),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", config["aws_secret_access_key"]),
            endpoint_url=config["endpoint_url"],
        )
    except KeyError as e:
        logger.error(f"Отсутствует ключ конфигурации: {e}")
        raise
    except BotoCoreError as e:
        logger.error(f"Ошибка подключения к S3: {e}")
        raise

# Функция скачивания файлов
def download_all_files(s3, bucket_name: str, local_root: Path):
    """Скачивает все файлы из указанного S3-бакета в локальную папку."""
    local_root.mkdir(parents=True, exist_ok=True)  # Убедимся, что папка существует
    continuation_token: Optional[str] = None  # Токен для постраничной загрузки

    try:
        while True:
            response = s3.list_objects_v2(Bucket=bucket_name, ContinuationToken=continuation_token) if continuation_token else s3.list_objects_v2(Bucket=bucket_name)

            if "Contents" not in response:
                logger.info("В бакете нет файлов для загрузки.")
                return

            for obj in response["Contents"]:
                file_key = obj["Key"]
                local_path = local_root / file_key

                local_path.parent.mkdir(parents=True, exist_ok=True)  # Создаем каталог, если его нет

                try:
                    s3.download_file(bucket_name, file_key, str(local_path))
                    logger.info(f"✅ Скачан: {file_key} → {local_path}")
                except BotoCoreError as e:
                    logger.error(f"Ошибка загрузки {file_key}: {e}")

            if response.get("IsTruncated"):  # Если список файлов обрезан, продолжаем загрузку
                continuation_token = response["NextContinuationToken"]
            else:
                break

    except BotoCoreError as e:
        logger.error(f"Ошибка при запросе списка файлов из S3: {e}")
    except NoCredentialsError:
        logger.error("Ошибка аутентификации: Проверьте AWS_ACCESS_KEY_ID и AWS_SECRET_ACCESS_KEY")

# Главная функция
def main():
    """Главная точка входа в программу."""
    config = load_config(CONFIG_PATH)
    s3 = create_s3_client(config)
    bucket_name = config.get("bucket_name", "koldyrkaevs3")  # Можно переопределить в YAML
    download_all_files(s3, bucket_name, LOCAL_ROOT)
    logger.info("🎉 Все файлы успешно загружены!")

if __name__ == "__main__":
    main()
