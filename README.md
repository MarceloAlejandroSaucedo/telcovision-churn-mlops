\# TelcoVision - Proyecto MLOps de Predicción de Churn


![CI Pipeline](https://github.com/MarceloAlejandroSaucedo/telcovision-churn-mlops/workflows/CI%20Pipeline%20-%20TelcoVision%20Churn/badge.svg)

## Proyecto ISTEA | Materia: Minería de datos II


\## 📋 Descripción del Proyecto



Pipeline reproducible de Machine Learning para predecir la rotación de clientes (churn) en la empresa ficticia \*\*TelcoVision\*\*, aplicando buenas prácticas de MLOps con versionado de datos y modelos.



\*\*Contexto:\*\* TelcoVision busca reducir la rotación de clientes mediante un modelo predictivo basado en datos de uso de servicios, información demográfica y métodos de pago.



\## 🎯 Objetivos



\- Construir un pipeline ML completamente reproducible

\- Aplicar control de versiones con DVC y Git

\- Trackear experimentos con MLflow

\- Implementar CI/CD con GitHub Actions

\- Predecir churn con métricas de alta calidad



\## 🛠️ Tecnologías Utilizadas



\- \*\*Python 3.11\*\* - Lenguaje principal

\- \*\*DVC\*\* - Versionado de datos y modelos

\- \*\*Git/GitHub\*\* - Control de versiones de código

\- \*\*DagsHub\*\* - Storage remoto y tracking

\- \*\*MLflow\*\* - Tracking de experimentos

\- \*\*scikit-learn\*\* - Machine Learning

\- \*\*Pandas/NumPy\*\* - Manipulación de datos



\## 📊 Dataset



\- \*\*Nombre:\*\* telco\_churn.csv

\- \*\*Registros:\*\* 10,000 clientes

\- \*\*Variables:\*\* 13 columnas (demográficas, servicios, churn)

\- \*\*Target:\*\* churn (1 = se da de baja, 0 = permanece)



\### Variables principales:

\- `customer\_id`: Identificador único

\- `age`: Edad del cliente

\- `gender`: Género (Male/Female)

\- `tenure\_months`: Meses como cliente

\- `monthly\_charges`: Cargos mensuales

\- `total\_charges`: Cargos totales

\- `contract\_type`: Tipo de contrato

\- `churn`: Variable objetivo



\## ⚙️ Requisitos Previos



Antes de comenzar, asegúrate de tener instalado:



\- Python 3.11+

\- Conda/Anaconda

\- Git

\- Cuenta en \[DagsHub](https://dagshub.com/)



\## 🚀 Instalación y Configuración



\### 1. Clonar el repositorio



git clone https://github.com/MarceloAlejandroSaucedo/telcovision-churn-mlops.git

cd telcovision-churn-mlops



\### 2. Crear entorno virtual con Conda



Crear entorno

conda create -n telcovision-mlops python=3.11 -y



Activar entorno

conda activate telcovision-mlops




\### 3. Instalar dependencias



pip install -r requirements.txt




\### 4. Configurar credenciales de DagsHub



Para descargar los datos versionados, necesitas configurar tu token de DagsHub:



1\. Ve a \[DagsHub Settings → Tokens](https://dagshub.com/user/settings/tokens)

2\. Genera un nuevo token con permisos de lectura

3\. Configura el remote DVC localmente:



dvc remote modify origin --local auth basic

dvc remote modify origin --local user TU\_USUARIO\_DAGSHUB

dvc remote modify origin --local password TU\_TOKEN\_DAGSHUB






\### 5. Descargar datos versionados



dvc pull






\### 6. Ejecutar el pipeline completo



dvc repro






\## 📁 Estructura del Proyecto

telcovision-churn-mlops/
├── data/
│   ├── raw/                  # Datos originales (versionado con DVC)
│   │   └── telco_churn.csv
│   └── processed/            # Datos procesados (versionado con DVC)
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       ├── y_test.csv
│       └── metadata.json
├── src/
│   ├── data_prep.py          # Script de preparación de datos
│   ├── train.py              # Script de entrenamiento del modelo
│   └── evaluate.py           # Script de evaluación avanzada (Etapa 7)
├── models/                   # Modelos entrenados (versionado con DVC)
│   ├── model.joblib
│   └── metrics.json
├── evaluation/               # Visualizaciones avanzadas (Etapa 7 Bonus)
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── classification_report.txt
│   ├── classification_report.json
│   └── advanced_metrics.json
├── .dvc/                     # Configuración de DVC
├── .github/
│   └── workflows/            # GitHub Actions CI/CD
├── params.yaml               # Parámetros configurables del pipeline
├── dvc.yaml                  # Definición del pipeline DVC
├── dvc.lock                  # Estado del pipeline (reproducibilidad)
├── DEPLOYMENT.md             # Estrategia de deployment
├── requirements.txt          # Dependencias Python
├── .gitignore                # Archivos ignorados por Git
└── README.md                 # Este archivo







\## 🔄 Pipeline de Trabajo

El proyecto implementa un pipeline reproducible con tres etapas principales:

### Etapa 1: Preparación de Datos (`prepare`)

**Script:** `src/data_prep.py`

**Funciones:**
- Carga del dataset raw
- Limpieza de datos (valores nulos, duplicados)
- Codificación de variables categóricas (LabelEncoder)
- División train/test (80/20) estratificada
- Escalado de variables numéricas (StandardScaler)
- Generación de datasets procesados

**Entradas:**
- `data/raw/telco_churn.csv`
- `params.yaml`

**Salidas:**
- `data/processed/X_train.csv`
- `data/processed/X_test.csv`
- `data/processed/y_train.csv`
- `data/processed/y_test.csv`
- `data/processed/metadata.json`

### Etapa 2: Entrenamiento del Modelo (`train`)

**Script:** `src/train.py`

**Funciones:**
- Carga de datos procesados
- Entrenamiento de modelo (Logistic Regression)
- Cálculo de métricas (accuracy, precision, recall, F1, ROC-AUC)
- Guardado del modelo entrenado
- Tracking con MLflow (opcional)

**Entradas:**
- `data/processed/X_train.csv`
- `data/processed/X_test.csv`
- `data/processed/y_train.csv`
- `data/processed/y_test.csv`
- `params.yaml`

**Salidas:**
- `models/model.joblib`
- `metrics.json`

### Etapa 3: Evaluación Avanzada (`evaluate`) ⭐ BONUS

**Script:** `src/evaluate.py`

**Funciones:**
- Generación de visualizaciones avanzadas
- Matriz de confusión
- Curva ROC
- Curva Precision-Recall
- Reportes de clasificación detallados
- Métricas adicionales

**Entradas:**
- `data/processed/X_test.csv`
- `data/processed/y_test.csv`
- `models/model.joblib`

**Salidas:**
- `evaluation/confusion_matrix.png`
- `evaluation/roc_curve.png`
- `evaluation/precision_recall_curve.png`
- `evaluation/classification_report.txt`
- `evaluation/classification_report.json`
- `evaluation/advanced_metrics.json`


\## 📈 Reproducibilidad



Para reproducir todo el pipeline desde cero:



Ejecutar todas las etapas

dvc repro



Ver el DAG del pipeline

dvc dag



Verificar estado

dvc status



Ver diferencias en parámetros

dvc params diff






\## 🔧 Configuración de Parámetros



Edita `params.yaml` para modificar hiperparámetros sin cambiar código:



data\_prep:

test\_size: 0.2

random\_state: 42

target\_column: churn



train:

model\_type: random\_forest

random\_forest:

n\_estimators: 100

max\_depth: 10

min\_samples\_split: 5

min\_samples\_leaf: 2

random\_state: 42






Después de modificar parámetros, ejecuta:



dvc repro






DVC detectará automáticamente los cambios y solo re-ejecutará las etapas necesarias.



### Experimentos Realizados

Se ejecutaron 3 experimentos variando hiperparámetros del RandomForestClassifier:

| Experimento | n_estimators | max_depth | min_samples_split | Test Accuracy | Test Recall | Test ROC-AUC |
|-------------|--------------|-----------|-------------------|---------------|-------------|--------------|
| **aided-spit (Baseline)** | 100 | 10 | 5 | 66.65% | 35.90% | 71.21% |
| **raked-skis (Alta potencia)** | 200 | 20 | 2 | 66.80% | **41.40%** | 70.64% |
| **milky-dops (Balanceado) ** | 150 | 15 | 10 | **67.20%** | 40.17% | 70.94% |

### Modelo Seleccionado

**Experimento: milky-dops (Balanceado)**

**Justificación:**
- Mejor accuracy general (67.20%)
- Recall competitivo (40.17%), solo 1.2 puntos menos que el mejor
- Mejor precision (56.92%) reduciendo falsas alarmas
- Balance óptimo entre todas las métricas para uso en producción

**Métricas detalladas del mejor modelo:**
- Test Accuracy: 0.6720
- Test Precision: 0.5692
- Test Recall: 0.4017
- Test F1-Score: 0.4710
- Test ROC-AUC: 0.7094

**Comparación completa:** Ver archivo `experimentos_comparacion.txt` o ejecutar `dvc exp show` para detalles completos.

**Reproducción de experimentos:**




\## 🧪 Experimentación



Para ejecutar experimentos con diferentes hiperparámetros:



1\. Modifica los valores en `params.yaml`

2\. Ejecuta `dvc repro`

3\. Las métricas se actualizarán automáticamente

4\. Compara resultados en DagsHub

5\. Haz commit de los cambios:



git add params.yaml dvc.lock models/metrics.json

git commit -m "exp: test n\_estimators=200"

git push


Ver todos los experimentos
dvc exp show

Aplicar un experimento específico
dvc exp apply milky-dops

Ver métricas
dvc metrics show


Los experimentos y artefactos están versionados con DVC y disponibles en DagsHub.

\## 🔗 Enlaces del Proyecto



\- \*\*Repositorio GitHub:\*\* \[telcovision-churn-mlops](https://github.com/MarceloAlejandroSaucedo/telcovision-churn-mlops)

\- \*\*Proyecto DagsHub:\*\* \[telcovision-churn-mlops](https://dagshub.com/MarceloAlejandroSaucedo/telcovision-churn-mlops)



\## 🚦 CI/CD con GitHub Actions



El proyecto incluye automatización con GitHub Actions que:



\- Verifica la reproducibilidad del pipeline

\- Ejecuta tests automáticos

\- Valida la calidad del código

\- Se ejecuta en cada push o pull request



\## 🐛 Resolución de Problemas



\### Error: "dvc pull" falla



Verificar configuración de remote

dvc remote list



Reconfigurar credenciales

dvc remote modify origin --local auth basic

dvc remote modify origin --local user TU\_USUARIO

dvc remote modify origin --local password TU\_TOKEN






\### Error: "dvc repro" no detecta cambios



Forzar re-ejecución de una etapa específica

dvc repro -f prepare



O de todo el pipeline

dvc repro -f






\### Error: Falta algún archivo



Descargar todos los archivos trackeados

dvc pull



Verificar status

dvc status


## Resultados Finales

### Modelo en Producción

**Algoritmo seleccionado:** Logistic Regression

| Métrica | Valor |
|---------|-------|
| Test Accuracy | 67.45% |
| Test Precision | 56.10% |
| Test Recall | 44.44% |
| Test F1-Score | 48.37% |
| Test ROC-AUC | 70.54% |
| PR-AUC | 54.24% |

### Experimentos Evaluados

Se probaron 3 enfoques diferentes mediante Pull Requests:

| Experimento | Accuracy | F1-Score | Decisión |
|-------------|----------|----------|----------|
| Logistic Regression | 67.45% | 48.37% | ✅ Seleccionado |
| RF Tuning | 66.85% | 47.17% | ❌ Descartado |
| Feature Engineering | 66.55% | 45.65% | ❌ Overfitting |

**Justificación:** Logistic Regression demostró el mejor balance entre performance y simplicidad, evitando overfitting.

### Visualizaciones

El pipeline genera automáticamente:
- Matriz de confusión para análisis de errores
- Curva ROC para evaluar discriminación del modelo
- Curva Precision-Recall para datasets desbalanceados
- Reportes de clasificación detallados

**Ver:** Carpeta `evaluation/` después de ejecutar `dvc repro`

## 🚀 Deployment

Para información sobre estrategia de deployment en producción, ver [DEPLOYMENT.md](DEPLOYMENT.md)

Incluye:
- Arquitectura propuesta (API REST vs Batch)
- Stack tecnológico recomendado
- Estrategia de monitoreo y reentrenamiento
- Estimación de costos



\## 👤 Autor



\*\*Marcelo Alejandro Saucedo\*\*

\*\*Daniel Alejandro Bastidas\*\*

\*\*Rosario Ratto\*\*

\- GitHub: \[@MarceloAlejandroSaucedo](https://github.com/MarceloAlejandroSaucedo)

\- Curso: Laboratorio de Minería de Datos II - ISTEA

\- Fecha: Octubre 2025

---

## 🚀 CI/CD Status

Este proyecto utiliza GitHub Actions para validar automáticamente cada cambio.



