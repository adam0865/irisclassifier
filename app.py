import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
model = RandomForestClassifier()
model.fit(X, y)

# Streamlit App
st.title("🌸 Klasifikasi Iris Flower")

st.markdown("Masukkan nilai fitur bunga di bawah ini:")

# Input sliders untuk fitur
sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 1.0)

# Buat dataframe dari input
input_data = pd.DataFrame(
    [[sepal_length, sepal_width, petal_length, petal_width]],
    columns=iris.feature_names
)

# Prediksi
prediction = model.predict(input_data)[0]
prediction_proba = model.predict_proba(input_data)

# Tampilkan hasil
st.subheader("Hasil Prediksi:")
st.write(f"🌼 **Jenis Iris:** {iris.target_names[prediction].capitalize()}")

st.subheader("Probabilitas:")
st.bar_chart(prediction_proba[0])
