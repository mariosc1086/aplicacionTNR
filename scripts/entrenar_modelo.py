from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import joblib
import pandas as pd
import numpy as np

# 1. Leer base
data_tnr = pd.read_csv("data/data_tnr.csv")

# 2. Renombrar columnas
data_tnr = data_tnr.rename(columns={
    "Dominio geográfico": "Geografico",
    "Prom. Cantidad de visitas": "Visitas",
    "Prom. Duracion_Min": "Duracion_Min",
    "Cant. Vivienda": "Cant_Vivienda"
})

# 3. Ordenar temporalmente
data_tnr = data_tnr.sort_values(
    by=["Conglomerado", "Año", "Meses"]
)

# 4. Crear variable objetivo
data_tnr["TNR_ALTA"] = np.where(
    data_tnr["TNR"] > 5,
    1,
    0
)

orden_meses = {
    "ene": 1, "feb": 2, "mar": 3, "abr": 4,
    "may": 5, "jun": 6, "jul": 7, "ago": 8,
    "sep": 9, "oct": 10, "nov": 11, "dic": 12
}

data_tnr["Mes_num"] = data_tnr["Meses"].map(orden_meses)

data_tnr = data_tnr.sort_values(
    by=["Conglomerado", "Año", "Mes_num"]
)

# 5. Crear TNR histórica del conglomerado
data_tnr["TNR_Historica_Cong"] = (
    data_tnr
    .groupby("Conglomerado")["TNR"]
    .transform(lambda x: x.shift().expanding().mean())
)

# 6. Crear TNR histórica del distrito
data_tnr = data_tnr.sort_values(
    by=["Distrito", "Año", "Mes_num"]
)

data_tnr["TNR_Historica_Distrito"] = (
    data_tnr
    .groupby("Distrito")["TNR"]
    .transform(lambda x: x.shift().expanding().mean())
)

# 7. Crear TNR histórica del departamento
data_tnr = data_tnr.sort_values(
    by=["Departamento", "Año", "Mes_num"]
)

data_tnr["TNR_Historica_Departamento"] = (
    data_tnr
    .groupby("Departamento")["TNR"]
    .transform(lambda x: x.shift().expanding().mean())
)

# 8. Crear indicadores de disponibilidad histórica
data_tnr["Tiene_Historia_Cong"] = np.where(
    data_tnr["TNR_Historica_Cong"].isna(), 0, 1
)

data_tnr["Tiene_Historia_Distrito"] = np.where(
    data_tnr["TNR_Historica_Distrito"].isna(), 0, 1
)

data_tnr["Tiene_Historia_Dep"] = np.where(
    data_tnr["TNR_Historica_Departamento"].isna(), 0, 1
)

# 9. Imputar históricos faltantes por jerarquía

# Conglomerado: si falta, usar distrito
data_tnr["TNR_Historica_Cong"] = data_tnr["TNR_Historica_Cong"].fillna(
    data_tnr["TNR_Historica_Distrito"]
)

# Si todavía falta, usar departamento
data_tnr["TNR_Historica_Cong"] = data_tnr["TNR_Historica_Cong"].fillna(
    data_tnr["TNR_Historica_Departamento"]
)

# Si todavía falta, usar promedio global
promedio_global = data_tnr["TNR"].mean()

data_tnr["TNR_Historica_Cong"] = data_tnr["TNR_Historica_Cong"].fillna(
    promedio_global
)

# Distrito: si falta, usar departamento
data_tnr["TNR_Historica_Distrito"] = data_tnr["TNR_Historica_Distrito"].fillna(
    data_tnr["TNR_Historica_Departamento"]
)

# Si todavía falta, usar promedio global
data_tnr["TNR_Historica_Distrito"] = data_tnr["TNR_Historica_Distrito"].fillna(
    promedio_global
)

# Departamento: si falta, usar promedio global
data_tnr["TNR_Historica_Departamento"] = data_tnr["TNR_Historica_Departamento"].fillna(
    promedio_global
)

features = [
    "Año",
    "Meses",
    "Departamento",
    "Estratos",
    "Geografico",
    "Visitas",
    "TNR_Historica_Cong",
    "TNR_Historica_Distrito",
    "TNR_Historica_Departamento",
    "TEM",
    "N_HOGAR"
]

target = "TNR_ALTA"

train = data_tnr[data_tnr["Año"] <= 2025].copy()
test = data_tnr[data_tnr["Año"] == 2026].copy()

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

cat_vars = [
    "Meses",
    "Departamento",
    "Estratos",
    "Geografico"
]

num_vars = [
    "Año",
    "Visitas",
    "TNR_Historica_Cong",
    "TNR_Historica_Distrito",
    "TNR_Historica_Departamento",
    "TEM",
    "N_HOGAR"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_vars),
        ("num", "passthrough", num_vars)
    ]
)

modelo_xgb = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="auc",
    random_state=123
)

pipeline_xgb = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("modelo", modelo_xgb)
    ]
)

pipeline_xgb.fit(X_train, y_train)

prob_test = pipeline_xgb.predict_proba(X_test)[:, 1]

threshold = 0.4292

pred_test = (prob_test >= threshold).astype(int)

resultados = {
    "Accuracy": accuracy_score(y_test, pred_test),
    "Recall": recall_score(y_test, pred_test),
    "Precision": precision_score(y_test, pred_test),
    "F1": f1_score(y_test, pred_test),
    "AUC": roc_auc_score(y_test, prob_test)
}

joblib.dump(pipeline_xgb, "models/pipeline_xgb_tnr.pkl")
joblib.dump(threshold, "models/threshold_tnr.pkl")


