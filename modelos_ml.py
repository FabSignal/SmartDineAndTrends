import os
import pickle
import pandas as pd
import numpy as np
import joblib
from scipy.special import softmax
import streamlit as st
import ast
from prophet import Prophet

# Diccionario que mapea las fechas a un valor numérico en meses (p.ej. Junio 2025 = 6, Diciembre 2025 = 12, etc.)
month_mapping = {
    'Junio 2025': 6,
    'Diciembre 2025': 12,
    'Junio 2026': 18,
    'Diciembre 2026': 24,
    'Junio 2027': 30,
    'Diciembre 2027': 36
}

# Función para cargar los modelos entrenados desde la carpeta correspondiente al estado
def load_models(state):
    model_dir = f'model_output_directory_{state.lower()}'
    
    # Verificar que la carpeta exista
    if not os.path.exists(model_dir):
        st.error(f"Ups, no se encontraron modelos para {state}. ¿Estás seguro de que seleccionaste el estado correcto?")
        return None
    
    models = {}
    
    # Leer todos los archivos .pkl en el directorio del estado
    for filename in os.listdir(model_dir):
        if filename.endswith(".pkl"):
            category = filename.replace("_prophet_model.pkl", "")
            model_path = os.path.join(model_dir, filename)
            
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
                # Almacenar el modelo en el diccionario
                models[category] = {'model': model, 'data': None}  # Aquí data podría estar, si se desea incluir los datos originales

    return models

# Función para realizar predicciones y calcular tasas de crecimiento
def predict_and_calculate_growth(state, months, growth_rates):
    models = load_models(state)
    if models is None:
        return None

    future_predictions = {}
    
    # Realizar predicciones para cada categoría
    for category, model_data in models.items():
        model = model_data['model']
        
        # Crear DataFrame futuro
        future = model.make_future_dataframe(periods=months, freq='M')
        forecast = model.predict(future)
        
        # Guardar predicción
        future_predictions[category] = forecast

    # Calcular tasas de crecimiento
    growth_results = {}
    for category, forecast in future_predictions.items():
        initial_value = forecast.loc[forecast['ds'] == forecast['ds'].min(), 'yhat'].values[0]
        final_value = forecast.loc[forecast['ds'] == forecast['ds'].max(), 'yhat'].values[0]
        growth_rate = ((final_value - initial_value) / initial_value) * 100
        
        # Aumentar tasa de crecimiento para categorías prioritarias
        if category in ['asian', 'vegan/vegetarian', 'seafood', 'coffee/tea culture', 'mediterranean']:
            growth_rate += 20  # Incremento para categorías prioritarias
        
        growth_results[category] = growth_rate

    # Crear resumen
    growth_summary = pd.DataFrame.from_dict(growth_results, orient='index', columns=['Growth Rate (%)'])
    growth_summary = growth_summary.sort_values(by='Growth Rate (%)', ascending=False)
    return growth_summary

# Función para convertir la columna 'categories' a listas de categorías (mismo proceso de preprocesamiento)
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

# Función para convertir las categorías a vectores binarios
def vector_binario_de_categorias(categorias, categoria_a_indice):
    vector = np.zeros(len(categoria_a_indice))  # Crear un vector de ceros
    for categoria in categorias:
        if categoria in categoria_a_indice:
            vector[categoria_a_indice[categoria]] = 1
    return vector

# Función para cargar los archivos desde una carpeta dada
def cargar_modelo_y_datos(estado):
    # Carpeta donde están los archivos para el estado
    carpeta = os.path.join('modelos_y_datos', estado)
    
    # Cargar el CSV
    df = pd.read_csv(os.path.join(carpeta, f'data_{estado}.csv'))
    
    # Cargar los objetos guardados (modelos y codificadores)
    classifier = joblib.load(os.path.join(carpeta, f'modelo_restaurant_lightgbm_{estado}.pkl'))
    name_encoder = joblib.load(os.path.join(carpeta, f'name_encoder_{estado}.pkl'))
    state_encoder = joblib.load(os.path.join(carpeta, f'state_encoder_{estado}.pkl'))
    city_encoder = joblib.load(os.path.join(carpeta, f'city_encoder_{estado}.pkl'))
    svd = joblib.load(os.path.join(carpeta, f'svd_transformer_{estado}.pkl'))
    categoria_a_indice = joblib.load(os.path.join(carpeta, f'categoria_a_indice_{estado}.pkl'))
    
    return df, classifier, name_encoder, state_encoder, city_encoder, svd, categoria_a_indice

# Configuración de Streamlit
st.title("🍽️ Plataforma de Predicción de Restaurantes 🍽️")

# Crear pestañas para separar las funcionalidades
tab1, tab2 = st.tabs(["Predicción de Categorías Emergentes", "Recomendación de Restaurantes"])

# Pestaña 1: Predicción de Categorías Emergentes
with tab1:
    st.header("🔮 Predicción de Categorías Emergentes de Restaurantes 🔮")
    st.sidebar.header("🚀 Parámetros de Entrada 🚀")
    state = st.sidebar.selectbox(
        "Selecciona un estado 🗺️", 
        ["florida", "california"],
        help="Elige el estado para obtener las predicciones más relevantes."
    )

    # Selección de fecha en el selectbox
    month_selection = st.sidebar.selectbox(
        "¿Hasta qué mes quieres predecir? 📅", 
        options=list(month_mapping.keys()),  # Usamos las fechas como opciones
        index=0,  # El valor predeterminado será 'Junio 2025'
        help="Elige un rango de meses para proyectar las predicciones."
    )

    # Obtener el valor numérico correspondiente a la fecha seleccionada
    months = month_mapping[month_selection]

    st.write(f"✨ Predicciones para el estado de *{state.capitalize()}* hasta **{month_selection}** (equivalente a {months} meses) ✨")

    # Botón para ejecutar predicción
    if st.sidebar.button("¡Hagamos las predicciones! 🎯"):
        growth_rates = {}
        results = predict_and_calculate_growth(state, months, growth_rates)

        if results is not None:
            # Mostrar las 5 categorías principales
            st.write("🔥 **Las 5 categorías que serán tendencia** 🔥")
            top_5 = results.head(5)
            for idx, (category, row) in enumerate(top_5.iterrows(), start=1):
                st.write(f"{idx}. **{category.capitalize()}**  *{row['Growth Rate (%)']:.2f}%*")
        
        # Si no hay resultados
        else:
            st.error("😱 ¡Algo salió mal! No pudimos obtener los resultados. Intenta con otro estado o verifica los modelos.")

# Pestaña 2: Recomendación de Restaurantes
with tab2:
    st.header("🍽️ Recomendación de Restaurantes 🍽️")
    st.markdown("""
        <h2>¿Dónde comemos? 🤔</h2>
        <h3>Bienvenido a tu guía de Restaurantes Personalizada 🍽️</h3>
    """, unsafe_allow_html=True)

    # Selección de la carpeta de datos (estado)
    states = ['florida', 'california']  # Nombres de los estados disponibles
    estado_seleccionado = st.selectbox('Selecciona un estado:', states)

    # Mapeo de nombres completos de los estados a las abreviaturas
    estado_abreviado = {
        'florida': 'FL',
        'california': 'CA'
    }

    # Obtener el estado seleccionado en formato abreviado
    estado_seleccionado_abreviado = estado_abreviado.get(estado_seleccionado)

    # Verificar si la abreviatura existe, y cargar los modelos y datos para ese estado
    if estado_seleccionado_abreviado:
        df_original, classifier, name_encoder, state_encoder, city_encoder, svd, categoria_a_indice = cargar_modelo_y_datos(estado_seleccionado_abreviado)
    else:
        st.error(f"El estado seleccionado ({estado_seleccionado}) no tiene una abreviatura válida.")
        st.stop()

    # Obtener las opciones únicas de 'state', 'city' y 'categories' de los datos originales
    cities = df_original['city'].unique()
    categories = convertir_a_lista(df_original, 'categories')['categories'].explode().unique()

    # Selección de la ciudad
    city = st.selectbox('Selecciona una ciudad:', cities)

    # Selección de las categorías (múltiples categorías)
    max_categories = 3  # Límite máximo de categorías
    selected_categories = st.multiselect(
        'Selecciona hasta 3 categorías:',
        categories.tolist(),
        max_selections=max_categories
    )

    # Validar que el usuario haya seleccionado al menos una categoría
    if not selected_categories:
        st.warning('Por favor, selecciona al menos una categoría para continuar.')

    # Validar que el usuario no haya seleccionado más de 3 categorías
    if len(selected_categories) > max_categories:
        st.warning(f'Puedes seleccionar un máximo de {max_categories} categorías. Has seleccionado {len(selected_categories)}.')

    # Crear el DataFrame con los datos seleccionados si hay categorías seleccionadas
    if selected_categories:
        df_nuevo = pd.DataFrame({
            'state': [estado_seleccionado],  # Usar el nombre completo del estado
            'city': [city],
            'categories': [selected_categories]  # Lista de categorías seleccionadas
        })

        # Preprocesar los datos nuevos
        df_nuevo = convertir_a_lista(df_nuevo, 'categories')

        # Codificar 'state' con la abreviatura correspondiente
        df_nuevo['state_encoded'] = state_encoder.transform([estado_seleccionado_abreviado])
        
        # Codificar 'city'
        df_nuevo['city_encoded'] = city_encoder.transform(df_nuevo['city'])

        # Convertir las categorías a vectores binarios
        df_nuevo['category_vector'] = df_nuevo['categories'].apply(vector_binario_de_categorias, args=(categoria_a_indice,))
        category_matrix = np.vstack(df_nuevo['category_vector'].values)

        # Crear las columnas de categorías como en el entrenamiento
        category_columns = [f'category_{i}' for i in range(category_matrix.shape[1])]
        category_df = pd.DataFrame(category_matrix, columns=category_columns)

        # Concatenar las columnas codificadas y los vectores binarios de categorías
        X_nuevo = pd.concat([df_nuevo[['state_encoded', 'city_encoded']], category_df], axis=1)

        # Aplicar la reducción de dimensionalidad (SVD) a las nuevas características de categorías
        X_nuevo_reducido = svd.transform(X_nuevo[category_columns])

        # Concatenar las características originales con las nuevas componentes de SVD
        X_nuevo_reducido = np.concatenate([X_nuevo[['state_encoded', 'city_encoded']].values, X_nuevo_reducido], axis=1)

        # Obtener las predicciones utilizando el modelo Booster
        y_pred_logits = classifier.predict(X_nuevo_reducido, raw_score=True)

        # Convertir los logits a probabilidades usando softmax
        y_pred_prob = softmax(y_pred_logits, axis=1)

        # Obtener los tres restaurantes con las mayores probabilidades
        top_3_indices = np.argsort(y_pred_prob[0])[-3:][::-1]

        # Decodificar los índices a los nombres originales
        top_3_restaurants = name_encoder.inverse_transform(top_3_indices)

        # Crear una lista para almacenar los resultados
        resultados = []

        for i in top_3_indices:
            restaurante = df_original.iloc[i]  # Obtener el restaurante según el índice
            categorias_restaurante = restaurante['categories']  # Obtener las categorías asociadas a este restaurante
            resultados.append({
                'restaurante': name_encoder.inverse_transform([i])[0],
                'categorias': categorias_restaurante
            })

        # Mostrar los resultados en Streamlit
        for i, res in enumerate(resultados):
            st.subheader(f"Restaurante {i+1}: {res['restaurante']}")
            st.write(f"Categorías: {res['categorias']}")
