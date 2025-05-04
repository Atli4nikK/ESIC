import streamlit as st
import numpy as np
import onnxruntime as ort
from PIL import Image
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
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(std=[0.3481, 0.3755, 0.3738], mean=[0.6722, 0.6039, 0.6150]),
    ])
    image = transform(image).unsqueeze(0).numpy()
    return image


def stable_softmax(x):
    e_x = np.exp(x - np.max(x))  # вычитаем максимум
    return e_x / np.sum(e_x)


# Функция для выполнения инференса
def predict(image, model, class_names):
    input_tensor = preprocess_image(image)
    outputs = model.run(None, {"input": input_tensor})
    logits = outputs[0][0]  # <-- извлекаем логиты для одного изображения
    probabilities = stable_softmax(logits)
    return probabilities


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
        probabilities = predict(image, model, class_names)
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
