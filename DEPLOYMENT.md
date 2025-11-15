\# Estrategia de Deployment - TelcoVision Churn



\## Resumen del Modelo



\- \*\*Algoritmo\*\*: Logistic Regression

\- \*\*Accuracy\*\*: 67.45%

\- \*\*F1-Score\*\*: 48.37%

\- \*\*ROC-AUC\*\*: 70.54%



\## Propuesta de Arquitectura



\### Opcion 1: API REST con FastAPI





from fastapi import FastAPI

import joblib

import pandas as pd



app = FastAPI()

model = joblib.load("models/model.joblib")



@app.post("/predict")

def predict(customer\_data: dict):

df = pd.DataFrame(\[customer\_data])

prediction = model.predict(df)

probability = model.predict\_proba(df)



return {

&nbsp;   "churn": int(prediction),

&nbsp;   "probability": float(probability)

}





\*\*Stack\*\*:

\- FastAPI + Uvicorn

\- Docker container

\- Deploy en AWS Lambda o GCP Cloud Run



\### Opcion 2: Batch Processing



Pipeline programado (cron job) que:

1\. Lee nuevos clientes desde DB

2\. Genera predicciones

3\. Guarda resultados en tabla de scoring

4\. Dispara alertas para clientes high-risk



\*\*Stack\*\*:

\- Apache Airflow para orquestacion

\- PostgreSQL para almacenamiento

\- Kubernetes para ejecucion



\## Monitoreo



\### Metricas a trackear:

\- \*\*Model Performance\*\*: Accuracy, F1, ROC-AUC

\- \*\*Data Drift\*\*: Distribucion de features vs datos de entrenamiento

\- \*\*Prediction Drift\*\*: Distribucion de predicciones



\### Herramientas:

\- Evidently AI para drift detection

\- Prometheus + Grafana para metricas

\- MLflow para registry de versiones



\## Actualizacion del Modelo



\### Trigger de reentrenamiento:

\- Performance degradation (F1 < threshold)

\- Data drift detectado

\- Cada N meses (calendario)



\### Proceso:

1\. DVC pipeline ejecuta reentrenamiento

2\. Validacion automatica en test set

3\. A/B testing en produccion (10% trafico)

4\. Rollout gradual si metricas mejoran



\## Consideraciones de Seguridad



\- API con autenticacion JWT

\- Rate limiting

\- Input validation estricta

\- Logs de auditorıa



\## Estimacion de Costos



\*\*AWS Lambda + API Gateway\*\*:

\- 1M requests/mes: ~$5 USD

\- Storage S3: ~$2 USD

\- \*\*Total\*\*: ~$10 USD/mes



\## Proximos Pasos



1\. Crear Dockerfile

2\. Implementar endpoint /predict

3\. Configurar CI/CD para deployment

4\. Setup de monitoreo

5\. Documentar runbook operacional

