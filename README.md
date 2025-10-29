\# TelcoVision - Proyecto MLOps de Predicción de Churn



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



text



\### 2. Crear entorno virtual con Conda



Crear entorno

conda create -n telcovision-mlops python=3.11 -y



Activar entorno

conda activate telcovision-mlops



text



\### 3. Instalar dependencias



pip install -r requirements.txt



text



\### 4. Configurar credenciales de DagsHub



Para descargar los datos versionados, necesitas configurar tu token de DagsHub:



1\. Ve a \[DagsHub Settings → Tokens](https://dagshub.com/user/settings/tokens)

2\. Genera un nuevo token con permisos de lectura

3\. Configura el remote DVC localmente:



dvc remote modify origin --local auth basic

dvc remote modify origin --local user TU\_USUARIO\_DAGSHUB

dvc remote modify origin --local password TU\_TOKEN\_DAGSHUB



text



\### 5. Descargar datos versionados



dvc pull



text



\### 6. Ejecutar el pipeline completo



dvc repro



text



\## 📁 Estructura del Proyecto



telcovision-churn-mlops/

├── data/

│ ├── raw/ # Datos originales (versionado con DVC)

│ │ └── telco\_churn.csv

│ └── processed/ # Datos procesados (versionado con DVC)

│ ├── X\_train.csv

│ ├── X\_test.csv

│ ├── y\_train.csv

│ ├── y\_test.csv

│ └── metadata.json

├── src/

│ ├── data\_prep.py # Script de preparación de datos

│ └── train.py # Script de entrenamiento del modelo

├── models/ # Modelos entrenados (versionado con DVC)

│ ├── model.joblib

│ └── metrics.json

├── .dvc/ # Configuración de DVC

├── .github/

│ └── workflows/ # GitHub Actions CI/CD

├── params.yaml # Parámetros configurables del pipeline

├── dvc.yaml # Definición del pipeline DVC

├── dvc.lock # Estado del pipeline (reproducibilidad)

├── requirements.txt # Dependencias Python

├── .gitignore # Archivos ignorados por Git

└── README.md # Este archivo



text



\## 🔄 Pipeline de Trabajo



El proyecto implementa un pipeline reproducible con dos etapas principales:



\### Etapa 1: Preparación de Datos (`prepare`)



\*\*Script:\*\* `src/data\_prep.py`



\*\*Funciones:\*\*

\- Carga del dataset raw

\- Limpieza de datos (valores nulos, duplicados)

\- Codificación de variables categóricas (LabelEncoder)

\- División train/test (80/20) estratificada

\- Escalado de variables numéricas (StandardScaler)

\- Generación de datasets procesados



\*\*Entradas:\*\*

\- `data/raw/telco\_churn.csv`

\- `params.yaml`



\*\*Salidas:\*\*

\- `data/processed/X\_train.csv`

\- `data/processed/X\_test.csv`

\- `data/processed/y\_train.csv`

\- `data/processed/y\_test.csv`

\- `data/processed/metadata.json`



\### Etapa 2: Entrenamiento del Modelo (`train`)



\*\*Script:\*\* `src/train.py`



\*\*Funciones:\*\*

\- Carga de datos procesados

\- Entrenamiento de modelo Random Forest

\- Cálculo de métricas (accuracy, precision, recall, F1, ROC-AUC)

\- Guardado del modelo entrenado

\- Tracking con MLflow (opcional)



\*\*Entradas:\*\*

\- `data/processed/X\_train.csv`

\- `data/processed/X\_test.csv`

\- `data/processed/y\_train.csv`

\- `data/processed/y\_test.csv`

\- `params.yaml`



\*\*Salidas:\*\*

\- `models/model.joblib`

\- `models/metrics.json`



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



text



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



text



Después de modificar parámetros, ejecuta:



dvc repro



text



DVC detectará automáticamente los cambios y solo re-ejecutará las etapas necesarias.



\## 📊 Métricas del Modelo



Las métricas se guardan en `models/metrics.json` y son trackeadas por DVC:



{

"train": {

"accuracy": 0.8524,

"precision": 0.8234,

"recall": 0.7012,

"f1\_score": 0.7573,

"roc\_auc": 0.9145

},

"test": {

"accuracy": 0.6665,

"precision": 0.5649,

"recall": 0.3590,

"f1\_score": 0.4390,

"roc\_auc": 0.7121

}

}



text



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



text



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



text



\### Error: "dvc repro" no detecta cambios



Forzar re-ejecución de una etapa específica

dvc repro -f prepare



O de todo el pipeline

dvc repro -f



text



\### Error: Falta algún archivo



Descargar todos los archivos trackeados

dvc pull



Verificar status

dvc status



text



\## 👤 Autor



\*\*Marcelo Alejandro Saucedo\*\*

\*\*Daniel Alejandro Bastidas\*\*

\*\*Rosario Ratto\*\*

\- GitHub: \[@MarceloAlejandroSaucedo](https://github.com/MarceloAlejandroSaucedo)

\- Curso: Laboratorio de Minería de Datos II - ISTEA

\- Fecha: Octubre 2025





