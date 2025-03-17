FROM python:3.9-slim

# Установка зависимостей
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Установка pip
RUN pip install --upgrade pip

# Установка зависимостей из requirements.txt
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

# Копирование файлов приложения
COPY . /app

# Команда для запуска приложения
CMD ["python", "checking.py"]
