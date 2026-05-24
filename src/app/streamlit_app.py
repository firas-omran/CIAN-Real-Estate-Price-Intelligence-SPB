"""Streamlit UI для RealEstate Price Explorer."""

import streamlit as st

st.set_page_config(
    page_title="CIAN Real Estate Price Intelligence",
    layout="wide",
)

st.title("CIAN Real Estate Price Intelligence")
st.markdown("ML-система для оценки стоимости квартир в Санкт-Петербурге по свежим объявлениям CIAN.")

st.sidebar.header("Параметры квартиры")

region = st.sidebar.selectbox("Регион", options=[77, 78, 50, 47, 23], format_func=lambda x: {
    77: "Москва", 78: "Санкт-Петербург", 50: "Московская обл.",
    47: "Ленинградская обл.", 23: "Краснодарский край",
}.get(x, str(x)))

rooms = st.sidebar.selectbox("Количество комнат", options=[-1, 1, 2, 3, 4, 5], format_func=lambda x: "Студия" if x == -1 else str(x))

area = st.sidebar.slider("Площадь (м2)", min_value=10, max_value=300, value=50)

building_type = st.sidebar.selectbox("Тип здания", options=[0, 1, 2, 3, 4, 5, 6], format_func=lambda x: {
    0: "Не указан", 1: "Другой", 2: "Панельный", 3: "Монолитный",
    4: "Кирпичный", 5: "Блочный", 6: "Деревянный",
}.get(x, str(x)))

if st.sidebar.button("Оценить стоимость"):
    st.info("Модель будет подключена в чекпоинте 3. Сейчас это заглушка.")
    st.metric("Предсказанная цена", "-- руб.")

st.markdown("---")
st.markdown("*Проект в рамках курса 'Архитектура ИИ' - ИТМО*")
