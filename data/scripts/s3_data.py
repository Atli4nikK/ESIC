import os
import boto3

# Твои креды
aws_access_key_id = "YCAJENQnEljQRBRBKMqbM4Jl1"
aws_secret_access_key = "YCMqKp43rXZVGwu3JWPbGZ2pd7t7g-PTwXVg0WQB"
endpoint_url = "https://storage.yandexcloud.net"

# Подключение к хранилищу
s3 = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    endpoint_url=endpoint_url,
)

# Имя бакета (замени на нужное)
bucket_name = "koldyrkaevs3"

# Локальная папка, куда всё скачивать
local_root = "./data/s3_downloaded"

# Функция для загрузки всех файлов из бакета
def download_all_files():
    continuation_token = None  # Токен для загрузки следующих частей файлов

    while True:
        # Запрашиваем список файлов (максимум 1000 за раз)
        if continuation_token:
            response = s3.list_objects_v2(Bucket=bucket_name, ContinuationToken=continuation_token)
        else:
            response = s3.list_objects_v2(Bucket=bucket_name)

        # Если файлы есть, качаем
        if "Contents" in response:
            for obj in response["Contents"]:
                file_key = obj["Key"]  # Полный путь файла в бакете (например, dataset/ambulance/image1.jpg)
                local_path = os.path.join(local_root, file_key)  # Локальный путь сохранения

                # Создаём локальную папку, если её нет
                os.makedirs(os.path.dirname(local_path), exist_ok=True)

                # Скачиваем файл
                s3.download_file(bucket_name, file_key, local_path)
                print(f"Скачан: {file_key} → {local_path}")

        # Проверяем, есть ли ещё файлы для загрузки
        if response.get("IsTruncated"):  # Если список обрезан, загружаем дальше
            continuation_token = response["NextContinuationToken"]
        else:
            break

# Запуск скачивания
download_all_files()
print("✅ Все файлы скачаны!")
