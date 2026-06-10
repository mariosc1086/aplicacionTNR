import streamlit as st
import pandas as pd
from scripts.prediccion import predecir_tnr

st.set_page_config(
    page_title="SIAT-TNR",
    page_icon="📊",
    layout="wide"
)

st.title("Sistema Inteligente de Alerta Temprana de TNR")
st.write("Aplicación para estimar la probabilidad de TNR alta por conglomerado.")

# Cargar base
data = pd.read_csv("data/data_tnr.csv")

# Renombrar columnas si es necesario
data = data.rename(columns={
    "Dominio geográfico": "Geografico",
    "Prom. Cantidad de visitas": "Visitas"
})

st.sidebar.header("Consulta del conglomerado")

departamento = st.sidebar.selectbox(
    "Departamento",
    sorted(data["Departamento"].dropna().unique())
)

provincia = st.sidebar.selectbox(
    "Provincia",
    sorted(data[data["Departamento"] == departamento]["Provincia"].dropna().unique())
)

distrito = st.sidebar.selectbox(
    "Distrito",
    sorted(data[
        (data["Departamento"] == departamento) &
        (data["Provincia"] == provincia)
    ]["Distrito"].dropna().unique())
)

conglomerado = st.sidebar.selectbox(
    "Conglomerado",
    sorted(data[
        (data["Departamento"] == departamento) &
        (data["Provincia"] == provincia) &
        (data["Distrito"] == distrito)
    ]["Conglomerado"].dropna().unique())
)

fila = data[
    (data["Departamento"] == departamento) &
    (data["Provincia"] == provincia) &
    (data["Distrito"] == distrito) &
    (data["Conglomerado"] == conglomerado)
].sort_values(["Año", "Meses"]).tail(1)

if st.sidebar.button("Calcular probabilidad"):

    if fila.empty:
        st.error("No se encontró información para el conglomerado seleccionado.")
    else:
        registro = fila.iloc[0]

        datos_modelo = {
            "Año": registro["Año"],
            "Meses": registro["Meses"],
            "Departamento": registro["Departamento"],
            "Estratos": registro["Estratos"],
            "Geografico": registro["Geografico"],
            "Visitas": registro["Visitas"],
            "TNR_Historica_Cong": registro["TNR_Historica_Cong"],
            "TNR_Historica_Distrito": registro["TNR_Historica_Distrito"],
            "TNR_Historica_Departamento": registro["TNR_Historica_Departamento"],
            "TEM": registro["TEM"],
            "N_HOGAR": registro["N_HOGAR"]
        }

        resultado = predecir_tnr(datos_modelo)

        st.subheader("Resultado del conglomerado")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Probabilidad TNR Alta",
            f"{resultado['probabilidad_porcentaje']}%"
        )

        col2.metric(
            "Nivel de riesgo",
            resultado["nivel_riesgo"]
        )

        col3.metric(
            "Clasificación",
            "TNR Alta" if resultado["clasificacion"] == 1 else "TNR No Alta"
        )

        st.markdown("---")

        st.write("### Información del conglomerado")

        st.dataframe(fila)

        st.write("### Variables usadas por el modelo")

        st.json(datos_modelo)