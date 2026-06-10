import joblib
import pandas as pd

# Cargar modelo y threshold
pipeline_xgb = joblib.load("models/pipeline_xgb_tnr.pkl")
threshold = joblib.load("models/threshold_tnr.pkl")


def clasificar_riesgo(probabilidad):
    if probabilidad < 0.30:
        return "Bajo"
    elif probabilidad < threshold:
        return "Medio"
    elif probabilidad < 0.70:
        return "Alto"
    else:
        return "Crítico"


def predecir_tnr(datos_conglomerado):
    """
    datos_conglomerado debe ser un diccionario con las variables del modelo.
    """

    df = pd.DataFrame([datos_conglomerado])

    probabilidad = pipeline_xgb.predict_proba(df)[:, 1][0]

    prediccion = int(probabilidad >= threshold)

    riesgo = clasificar_riesgo(probabilidad)

    resultado = {
        "probabilidad_tnr_alta": round(probabilidad, 4),
        "probabilidad_porcentaje": round(probabilidad * 100, 2),
        "clasificacion": prediccion,
        "nivel_riesgo": riesgo
    }

    return resultado