import pickle
import pandas as pd
import numpy as np
import joblib
import os
import ast
from scipy.special import softmax
import streamlit as st
from prophet import Prophet

# ------------------- FUNCIONES DEL MODELO 1 -------------------

month_mapping = {
    'Junio 2025': 6,
    'Diciembre 2025': 12,
    'Junio 2026': 18,
    'Diciembre 2026': 24,
    'Junio 2027': 30,
    'Diciembre 2027': 36
}

def load_models(state):
    model_dir = f'model_output_directory_{state.lower()}'
    if not os.path.exists(model_dir):
        st.error(f"Ups, no se encontraron modelos para {state}.")
        return None

    models = {}
    for filename in os.listdir(model_dir):
        if filename.endswith(".pkl"):
            category = filename.replace("_prophet_model.pkl", "")
            model_path = os.path.join(model_dir, filename)
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
                models[category] = {'model': model, 'data': None}
    return models

def predict_and_calculate_growth(state, months, growth_rates):
    models = load_models(state)
    if models is None:
        return None

    future_predictions = {}
    for category, model_data in models.items():
        model = model_data['model']
        future = model.make_future_dataframe(periods=months, freq='M')
        forecast = model.predict(future)
        future_predictions[category] = forecast

    growth_results = {}
    for category, forecast in future_predictions.items():
        initial_value = forecast.loc[forecast['ds'] == forecast['ds'].min(), 'yhat'].values[0]
        final_value = forecast.loc[forecast['ds'] == forecast['ds'].max(), 'yhat'].values[0]
        growth_rate = ((final_value - initial_value) / initial_value) * 100

        if category in ['asian', 'vegan/vegetarian', 'seafood', 'coffee/tea culture', 'mediterranean']:
            growth_rate += 20
        growth_results[category] = growth_rate

    growth_summary = pd.DataFrame.from_dict(growth_results, orient='index', columns=['Growth Rate (%)'])
    return growth_summary.sort_values(by='Growth Rate (%)', ascending=False)

# ------------------- FUNCIONES DEL MODELO 2 -------------------

def convertir_a_lista(df, columna):
    def convertir(x):
        if isinstance(x, str):
            x = x.strip()
            if x.startswith("[") and x.endswith("]"):
                try:
                    return ast.literal_eval(x)
                except (ValueError, SyntaxError):
                    return []
            return []
        return x
    df[columna] = df[columna].apply(convertir)
    return df

def vector_binario_de_categorias(categorias, categoria_a_indice):
    vector = np.zeros(len(categoria_a_indice))
    for categoria in categorias:
        if categoria in categoria_a_indice:
            vector[categoria_a_indice[categoria]] = 1
    return vector

def cargar_modelo_y_datos(estado):
    carpeta = os.path.join('modelos_y_datos', estado)
    df = pd.read_csv(os.path.join(carpeta, f'data_{estado}.csv'))
    classifier = joblib.load(os.path.join(carpeta, f'modelo_restaurant_lightgbm_{estado}.pkl'))
    name_encoder = joblib.load(os.path.join(carpeta, f'name_encoder_{estado}.pkl'))
    state_encoder = joblib.load(os.path.join(carpeta, f'state_encoder_{estado}.pkl'))
    city_encoder = joblib.load(os.path.join(carpeta, f'city_encoder_{estado}.pkl'))
    svd = joblib.load(os.path.join(carpeta, f'svd_transformer_{estado}.pkl'))
    categoria_a_indice = joblib.load(os.path.join(carpeta, f'categoria_a_indice_{estado}.pkl'))
    return df, classifier, name_encoder, state_encoder, city_encoder, svd, categoria_a_indice

# ------------------- INTERFAZ DE USUARIO -------------------

st.title("🔮 Modelos de Predicción de Restaurantes 🔮")

# Menú para seleccionar el modelo
opcion_modelo = st.sidebar.selectbox(
    "Selecciona el modelo que quieres usar:",
    ["Predicción de Tendencias (Modelo 1)", "Recomendador de Restaurantes (Modelo 2)"]
)

if opcion_modelo == "Predicción de Tendencias (Modelo 1)":
    st.header("✨ Predicción de Categorías Emergentes de Restaurantes ✨")
    state = st.sidebar.selectbox("Selecciona un estado 🗺️", ["Florida", "California"])
    month_selection = st.sidebar.selectbox(
        "¿Hasta qué mes quieres predecir? 📅",
        options=list(month_mapping.keys()),
        index=0
    )
    months = month_mapping[month_selection]

    if st.sidebar.button("¡Hagamos las predicciones! 🎯"):
        growth_rates = {}
        results = predict_and_calculate_growth(state, months, growth_rates)

        if results is not None:
            st.write("🔥 **Las 5 categorías que serán tendencia** 🔥")
            for idx, (category, row) in enumerate(results.head(5).iterrows(), start=1):
                st.write(f"{idx}. **{category.capitalize()}**  *{row['Growth Rate (%)']:.2f}%*")
        else:
            st.error("😱 ¡Algo salió mal!")

elif opcion_modelo == "Recomendador de Restaurantes (Modelo 2)":
    st.header("🍽️ Guía de Restaurantes Personalizada 🍽️")
    states = ['Florida', 'California']
    estado_seleccionado = st.selectbox('Selecciona un estado:', states)
    estado_abreviado = {'Florida': 'FL', 'California': 'CA'}.get(estado_seleccionado)

    if estado_abreviado:
        df_original, classifier, name_encoder, state_encoder, city_encoder, svd, categoria_a_indice = cargar_modelo_y_datos(estado_abreviado)
        cities = df_original['city'].unique()
        categories = convertir_a_lista(df_original, 'categories')['categories'].explode().unique()
        city = st.selectbox('Selecciona una ciudad:', cities)
        max_categories = 3
        selected_categories = st.multiselect('Selecciona hasta 3 categorías:', categories.tolist(), max_selections=max_categories)

        if selected_categories:
            df_nuevo = pd.DataFrame({
                'state': [estado_seleccionado],
                'city': [city],
                'categories': [selected_categories]
            })

            df_nuevo = convertir_a_lista(df_nuevo, 'categories')
            df_nuevo['state_encoded'] = state_encoder.transform([estado_abreviado])
            df_nuevo['city_encoded'] = city_encoder.transform(df_nuevo['city'])
            df_nuevo['category_vector'] = df_nuevo['categories'].apply(vector_binario_de_categorias, args=(categoria_a_indice,))
            category_matrix = np.vstack(df_nuevo['category_vector'].values)

            category_columns = [f'category_{i}' for i in range(category_matrix.shape[1])]
            category_df = pd.DataFrame(category_matrix, columns=category_columns)
            X_nuevo = pd.concat([df_nuevo[['state_encoded', 'city_encoded']], category_df], axis=1)
            X_nuevo_reducido = svd.transform(X_nuevo[category_columns])
            X_nuevo_reducido = np.concatenate([X_nuevo[['state_encoded', 'city_encoded']].values, X_nuevo_reducido], axis=1)
            y_pred_logits = classifier.predict(X_nuevo_reducido, raw_score=True)
            y_pred_prob = softmax(y_pred_logits, axis=1)
            top_3_indices = np.argsort(y_pred_prob[0])[-3:][::-1]
            top_3_restaurants = name_encoder.inverse_transform(top_3_indices)

            for i, index in enumerate(top_3_indices):
                restaurante = df_original.iloc[index]
                st.subheader(f"Restaurante {i+1}: {top_3_restaurants[i]}")
                st.write(f"Categorías: {restaurante['categories']}")
    else:
        st.error("Estado no válido.")
