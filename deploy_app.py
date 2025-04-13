import streamlit as st
import pickle
import pandas as pd

# Cargar modelos entrenados
with open('random_forest.pkl', 'rb') as rf:
    rf_model = pickle.load(rf)

with open('support_vector_machine.pkl', 'rb') as sv:
    svc_model = pickle.load(sv)

with open('logistic_regression.pkl', 'rb') as lg:
    logreg_model = pickle.load(lg)

with open('decision_tree.pkl', 'rb') as dt:
    dt_model = pickle.load(dt)

with open('k-nearest_neighbors.pkl', 'rb') as knn:
    knn_model = pickle.load(knn)

# Cargar preprocesador
with open('preprocesador.pkl', 'rb') as f:
    preprocesador = pickle.load(f)

# Clasificación
def classify(pred):
    return 'Posible enfermedad cardíaca' if pred == 1 else 'No tiene enfermedad cardíaca'

# Sidebar para parámetros del usuario
def user_input_parameters():
    st.sidebar.header("Datos del Usuario")
    user_input = {
        'Gender': st.sidebar.selectbox('Género', ['Female', 'Male']),
        'Smoking': st.sidebar.selectbox('Fuma', ['Current', 'Former', 'Never']),
        'Alcohol Intake': st.sidebar.selectbox('Consumo de Alcohol', ['Heavy', 'Moderate', 'None']),
        'Family History': st.sidebar.selectbox('Antecedentes Familiares', ['Yes', 'No']),
        'Chest Pain Type': st.sidebar.selectbox('Tipo de Dolor en el Pecho', ['Asymptomatic', 'Atypical Angina', 'Non-anginal Pain', 'Typical Angina']),
        'Age': st.sidebar.slider('Edad', 20, 120, 50),
        'Cholesterol': st.sidebar.slider('Colesterol (mg/dL)', 60, 500, 200),
        'Blood Pressure': st.sidebar.slider('Presión Arterial (mmHg)', 70, 300, 120),
        'Heart Rate': st.sidebar.slider('Frecuencia Cardíaca (bpm)', 40, 200, 80),
        'Stress Level': st.sidebar.slider('Nivel de Estrés (1 a 10)', 1, 10, 5),
        'Blood Sugar': st.sidebar.slider('Azúcar en Sangre (mg/dL)', 50, 300, 100)
    }
    return pd.DataFrame([user_input])

# App principal
def main():
    st.set_page_config(page_title="Predicción Cardíaca", layout="centered")

    st.title("Predicción de Enfermedad Cardíaca Grupo 1 Talleres")

    st.markdown("""
    Esta herramienta utiliza modelos de Machine Learning para predecir si una persona podría tener
    una enfermedad cardíaca, basándose en distintos parámetros médicos. Ingrese los datos a continuación en la barra lateral:
    """)

    df_usuario = user_input_parameters()

    st.subheader("Datos ingresados:")
    st.dataframe(df_usuario, use_container_width=True)

    modelo_seleccionado = st.selectbox(
        'Selecciona el modelo de predicción',
        ['Random Forest', 'SVM', 'Logistic Regression', 'Decision Tree', 'KNN']
    )

    if st.button('Predecir'):
        datos_procesados = preprocesador.transform(df_usuario)

        if modelo_seleccionado == 'Random Forest':
            resultado = rf_model.predict(datos_procesados)
        elif modelo_seleccionado == 'SVM':
            resultado = svc_model.predict(datos_procesados)
        elif modelo_seleccionado == 'Logistic Regression':
            resultado = logreg_model.predict(datos_procesados)
        elif modelo_seleccionado == 'Decision Tree':
            resultado = dt_model.predict(datos_procesados)
        elif modelo_seleccionado == 'KNN':
            resultado = knn_model.predict(datos_procesados)

        pred = classify(resultado[0])
        st.success(pred)

if __name__ == '__main__':
    main()
