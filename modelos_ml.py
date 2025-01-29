import pickle
import pandas as pd
import numpy as np
import joblib
import os
import ast
from scipy.special import softmax
import streamlit as st
from prophet import Prophet

# =================== CONFIGURACIÓN ===================
st.set_page_config(page_title="Modelos de Restaurantes", layout="wide")

month_mapping = {
    'Junio 2025': 6,
    'Diciembre 2025': 12,
    'Junio 2026': 18,
    'Diciembre 2026': 24,
    'Junio 2027': 30,
    'Diciembre 2027': 36
}

# =================== FUNCIONES DEL MODELO 1 ===================
def load_models(state):
    model_dir = f'model_output_directory_{state.lower()}'
    if not os.path.exists(model_dir):
        st.error(f"No se encontraron modelos para {state}.")
        return None
    
    models = {}
    for filename in os.listdir(model_dir):
        if filename.endswith(".pkl"):
            category = filename.replace("_prophet_model.pkl", "")
            with open(os.path.join(model_dir, filename), 'rb') as file:
                models[category] = pickle.load(file)
    return models

def predict_and_calculate_growth(state, months):
    models = load_models(state)
    if models is None:
        return None
    
    growth_results = {}
    for category, model in models.items():
        future = model.make_future_dataframe(periods=months, freq='M')
        forecast = model.predict(future)
        initial_value, final_value = forecast.iloc[0]['yhat'], forecast.iloc[-1]['yhat']
        growth_rate = ((final_value - initial_value) / initial_value) * 100
        if category in ['asian', 'vegan/vegetarian', 'seafood', 'coffee/tea culture', 'mediterranean']:
            growth_rate += 20
        growth_results[category] = growth_rate

    return sorted(growth_results.items(), key=lambda x: x[1], reverse=True)  # Retorna lista ordenada

# =================== FUNCIONES DEL MODELO 2 ===================
def convertir_a_lista(df, columna):
    df[columna] = df[columna].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else [])
    return df

def vector_binario_de_categorias(categorias, categoria_a_indice):
    vector = np.zeros(len(categoria_a_indice))
    for categoria in categorias:
        if categoria in categoria_a_indice:
            vector[categoria_a_indice[categoria]] = 1
    return vector

def cargar_modelo_y_datos(estado):
    carpeta = os.path.join('modelos_y_datos', estado)
    try:
        df = pd.read_csv(os.path.join(carpeta, f'data_{estado}.csv'))
        classifier = joblib.load(os.path.join(carpeta, f'modelo_restaurant_lightgbm_{estado}.pkl'))
        name_encoder = joblib.load(os.path.join(carpeta, f'name_encoder_{estado}.pkl'))
        state_encoder = joblib.load(os.path.join(carpeta, f'state_encoder_{estado}.pkl'))
        city_encoder = joblib.load(os.path.join(carpeta, f'city_encoder_{estado}.pkl'))
        svd = joblib.load(os.path.join(carpeta, f'svd_transformer_{estado}.pkl'))
        categoria_a_indice = joblib.load(os.path.join(carpeta, f'categoria_a_indice_{estado}.pkl'))
        return df, classifier, name_encoder, state_encoder, city_encoder, svd, categoria_a_indice
    except Exception as e:
        st.error(f"Error al cargar modelos y datos: {e}")
        return None

# =================== INTERFAZ DE USUARIO ===================
st.title(" Modelos Avanzados para Restaurantes:") 
st.title(" Análisis, Predicción y Recomendación")

# Crear pestañas para dividir las funcionalidades
tab1, tab2 = st.tabs(["📈 Predicción de Tendencias de Categorías", "🍴 Recomendación de Restaurantes"])

with tab1:
    st.header("🔍 Explorando el Futuro de la Gastronomía: Predicción de Reviews por Categoría ")
    state = st.selectbox("Selecciona un estado 🗺️", ["Florida", "California"])
    month_selection = st.selectbox("¿Hasta qué mes quieres predecir? 📅", list(month_mapping.keys()))
    
    if st.button("¡Predecir Tendencias! 🎯"):
        results = predict_and_calculate_growth(state, month_mapping[month_selection])
        if results:
            top_categories = results[:5]  # Tomamos las 5 principales categorías
            st.write("🔥 **Top 5 Categorías en Crecimiento** 🔥")
            for idx, (category, _) in enumerate(top_categories, 1):
                st.write(f"**{idx}. {category.capitalize()}**")
        else:
            st.error("😱 No se pudieron generar predicciones.")

with tab2:
    st.header("🍽️ Guía de Restaurantes Personalizada 🍽️")
    states = {"Florida": "FL", "California": "CA"}
    estado_seleccionado = st.selectbox("Selecciona un estado:", list(states.keys()))
    estado_abreviado = states[estado_seleccionado]

    datos = cargar_modelo_y_datos(estado_abreviado)
    if datos:
        df_original, classifier, name_encoder, state_encoder, city_encoder, svd, categoria_a_indice = datos
        cities = df_original['city'].unique()
        categories = convertir_a_lista(df_original, 'categories')['categories'].explode().unique()
        city = st.selectbox("Selecciona una ciudad:", cities)
        selected_categories = st.multiselect("Selecciona hasta 3 categorías:", categories.tolist(), max_selections=3)
        
        if selected_categories:
            df_nuevo = pd.DataFrame({'state': [estado_seleccionado], 'city': [city], 'categories': [selected_categories]})
            df_nuevo = convertir_a_lista(df_nuevo, 'categories')
            df_nuevo['state_encoded'] = state_encoder.transform([estado_abreviado])
            df_nuevo['city_encoded'] = city_encoder.transform(df_nuevo['city'])
            df_nuevo['category_vector'] = df_nuevo['categories'].apply(lambda x: vector_binario_de_categorias(x, categoria_a_indice))
            category_matrix = np.vstack(df_nuevo['category_vector'].values)
            X_nuevo = np.concatenate([df_nuevo[['state_encoded', 'city_encoded']].values, svd.transform(category_matrix)], axis=1)
            y_pred_prob = softmax(classifier.predict(X_nuevo, raw_score=True), axis=1)
            top_3_indices = np.argsort(y_pred_prob[0])[-3:][::-1]
            top_3_restaurants = name_encoder.inverse_transform(top_3_indices)
            
            st.subheader("🍽️ Restaurantes Recomendados:")
            for i, nombre in enumerate(top_3_restaurants, 1):
                st.write(f"**{i}. {nombre}**")
