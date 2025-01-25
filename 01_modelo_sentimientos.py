import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import pickle
import os

# Directorio donde se encuentran los modelos

model_dir = os.path.join(os.path.dirname(__file__), 'models')


def cargar_modelo():
    """Carga el vectorizador y el modelo desde un archivo .pkl predeterminado."""
    vectorizador = pickle.load(open(os.path.join(model_dir, "sentiment_vectorizer.pkl"), 'rb'))
    modelo = pickle.load(open(os.path.join(model_dir, "sentiment_model.pkl"), 'rb'))
    return vectorizador, modelo

# Crear la interfaz de usuario de Streamlit
st.title("🔍 Análisis de Sentimientos de Texto")
st.write("💬 Ingresa un texto y descubre si es positivo, negativo o neutral.")

# Obtener el texto del usuario
texto = st.text_area("✏️ Ingrese el texto aquí:")

# Botón para realizar la predicción
if st.button("🔎 Analizar Sentimiento"):
    if texto.strip():
        vectorizador, modelo = cargar_modelo()
        texto_vectorizado = vectorizador.transform([texto])
        prediccion = modelo.predict(texto_vectorizado)[0]
        
        if prediccion == 'positivo':
            st.success("😊 El sentimiento del texto es: **Positivo**")
        elif prediccion == 'negativo':
            st.error("😞 El sentimiento del texto es: **Negativo**")
        else:
            st.warning("😐 El sentimiento del texto es: **Neutral**")
    else:
        st.warning("⚠️ Por favor, ingrese un texto para analizar.")

# Pie de página atractivo
st.markdown("---")
st.markdown("🚀 **¡Explora el poder del análisis de sentimientos con IA!**")
