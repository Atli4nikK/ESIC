import streamlit as st
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import pandas as pd
import os
import hashlib
from datetime import datetime

# Пути
FEEDBACK_FILE = "feedback.csv"
IMAGE_FOLDER = "feedback_images"

# Создаем папку для изображений (если её нет)
os.makedirs(IMAGE_FOLDER, exist_ok=True)


# Функция для хеширования изображений (SHA256)
def get_image_hash(image):
    image_bytes = image.tobytes()
    return hashlib.sha256(image_bytes).hexdigest()


# Загрузка ONNX модели
@st.cache_resource
def load_model():
    return ort.InferenceSession("model.onnx")


# Функция для предобработки изображения
def preprocess_image(image):
    image = image.convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(std=[0.3481, 0.3755, 0.3738], mean=[0.6722, 0.6039, 0.6150]),
    ])
    image = transform(image).unsqueeze(0).numpy()
    return image


# Стабильный softmax
def stable_softmax(x):
    x = x - np.max(x)
    e_x = np.exp(x)
    sum_e_x = np.sum(e_x)
    if sum_e_x == 0 or np.isnan(sum_e_x):
        return np.full_like(x, fill_value=1.0 / len(x))  # безопасный fallback
    return e_x / sum_e_x


# Основная безопасная функция предсказания
def safe_predict(image: Image.Image, model, input_name="input"):
    try:
        # Обработка EXIF-ориентации
        image = ImageOps.exif_transpose(image)

        # Приведение к RGB
        image = image.convert("RGB")

        # Преобразования
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(std=[0.3481, 0.3755, 0.3738], mean=[0.6722, 0.6039, 0.6150]),
        ])

        tensor = transform(image).unsqueeze(0).numpy()  # [1, 3, H, W]

        # Валидация формы
        if tensor.shape[1] != 3:
            st.warning(f"❗ Неверное число каналов: {tensor.shape[1]} (ожидалось 3)")
            return None

        # Прогон через модель
        outputs = model.run(None, {input_name: tensor})
        logits = outputs[0][0]  # [num_classes]

        probabilities = stable_softmax(logits)

        # Проверка NaN
        if np.any(np.isnan(probabilities)):
            st.warning("❗ Предсказание содержит NaN. Что-то пошло не так.")
            return None

        return probabilities

    except Exception as e:
        st.error(f"🔥 Ошибка предсказания: {str(e)}")
        return None


# Функция для сохранения обратной связи с изображением
def save_feedback(image, predicted_class, correct_class, probabilities):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Получаем уникальный хэш изображения
    image_hash = get_image_hash(image)
    image_path = os.path.join(IMAGE_FOLDER, f"{image_hash}.png")

    # Сохраняем изображение, если оно ещё не сохранено
    if not os.path.exists(image_path):
        image.save(image_path)

    # Записываем путь к изображению в CSV
    feedback_data = pd.DataFrame([{
        "timestamp": timestamp,
        "image_path": image_path,
        "predicted_class": predicted_class,
        "correct_class": correct_class,
        "probabilities": probabilities
    }])

    if os.path.exists(FEEDBACK_FILE):
        feedback_data.to_csv(FEEDBACK_FILE, mode="a", header=False, index=False)
    else:
        feedback_data.to_csv(FEEDBACK_FILE, index=False)


# Интерфейс Streamlit
def main():
    st.set_page_config(page_title="ESIC", page_icon="🚑🚒🚓")
    st.title("🚑🚒🚓 Emergency Services Image Classifier")
    st.write("Загрузите изображение и получите предсказание")

    # Названия классов
    class_names = ["Ambulance", "Fire", "Police"]

    # Загружаем модель ONNX
    model = load_model()

    # Загрузка изображения
    uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Загруженное изображение", use_container_width=True)

        # Запуск предсказания
        st.write("🔍 **Предсказания модели:**")
        probabilities = safe_predict(image, model, input_name="input")
        if probabilities is None:
            st.stop()  # останавливаем выполнение
        predicted_class = class_names[np.argmax(probabilities)]

        # Отображаем вероятности в виде столбиков
        for class_name, probability in zip(class_names, probabilities):
            st.write(f"{class_name}: {probability:.2%}")
            st.progress(float(probability))

        # Вывод предсказанного класса
        st.subheader(f"🏷️ Модель считает, что это: **{predicted_class}**")

        # Кнопки подтверждения/опровержения
        col1, col2 = st.columns(2)

        with col1:
            if st.button("✅ Верно"):
                st.success("Спасибо! Ваш отклик сохранен.")
                save_feedback(image, predicted_class, predicted_class, probabilities.tolist())

        with col2:
            if st.button("❌ Неверно"):
                st.session_state.feedback_mode = True  # Включаем режим исправления
                st.session_state.feedback_submitted = False  # Разрешаем отправку

        # Если пользователь нажал "Неверно", показываем выбор класса
        if st.session_state.get("feedback_mode", False) and not st.session_state.get("feedback_submitted", False):
            st.session_state.correct_class = st.selectbox("Выберите правильный класс:", class_names)

            if st.button("📩 Отправить корректный ответ"):
                save_feedback(image, predicted_class, st.session_state.correct_class, probabilities.tolist())
                st.session_state.feedback_submitted = True
                st.success(f"Спасибо! Записали правильный класс: {st.session_state.correct_class}")
                st.session_state.feedback_mode = False  # Выключаем режим исправления


if __name__ == '__main__':
    main()
