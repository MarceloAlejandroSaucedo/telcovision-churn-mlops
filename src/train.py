# src/train.py
"""
Script de entrenamiento para el proyecto TelcoVision
Entrena modelo Random Forest y calcula métricas de clasificación
Integra con MLflow para tracking de experimentos
"""

import pandas as pd
import numpy as np
import yaml
import json
import os
import argparse
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Configuración de MLflow
USE_MLFLOW = False
try:
    import mlflow
    import mlflow.sklearn
    if os.getenv('MLFLOW_TRACKING_URI'):
        USE_MLFLOW = True
        print(" MLflow habilitado para tracking")
except Exception as e:
    print(f"  MLflow no disponible: {e}")
    USE_MLFLOW = False

def load_params(params_path='params.yaml'):
    """Carga los parámetros desde params.yaml"""
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)
    return params

def load_processed_data(data_dir='data/processed'):
    """Carga los datos procesados"""
    print(f"\n Cargando datos procesados desde: {data_dir}")
    
    X_train = pd.read_csv(f'{data_dir}/X_train.csv')
    X_test = pd.read_csv(f'{data_dir}/X_test.csv')
    y_train = pd.read_csv(f'{data_dir}/y_train.csv').squeeze()
    y_test = pd.read_csv(f'{data_dir}/y_test.csv').squeeze()
    
    print(f"  ✓ X_train: {X_train.shape}")
    print(f"  ✓ X_test: {X_test.shape}")
    print(f"  ✓ y_train: {y_train.shape}")
    print(f"  ✓ y_test: {y_test.shape}")
    print(" Datos cargados exitosamente")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, params):
    """Entrena el modelo Random Forest"""
    print("\n Iniciando entrenamiento del modelo...")
    
    model_params = params['train']['random_forest']
    
    model = RandomForestClassifier(
        n_estimators=model_params['n_estimators'],
        max_depth=model_params['max_depth'],
        min_samples_split=model_params['min_samples_split'],
        min_samples_leaf=model_params['min_samples_leaf'],
        random_state=model_params['random_state'],
        n_jobs=-1,
        verbose=0
    )
    
    print(f"  Parámetros del modelo:")
    for key, value in model_params.items():
        print(f"    - {key}: {value}")
    
    model.fit(X_train, y_train)
    print(" Modelo entrenado exitosamente")
    
    return model, model_params

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evalúa el modelo y calcula métricas"""
    print("\n Evaluando modelo...")
    
    # Predicciones
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Métricas en train
    train_metrics = {
        'accuracy': accuracy_score(y_train, y_train_pred),
        'precision': precision_score(y_train, y_train_pred, zero_division=0),
        'recall': recall_score(y_train, y_train_pred, zero_division=0),
        'f1_score': f1_score(y_train, y_train_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    }
    
    # Métricas en test
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred, zero_division=0),
        'recall': recall_score(y_test, y_test_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_test_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_test_proba)
    }
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_test_pred)
    
    print("\n   Métricas en TRAIN:")
    for metric, value in train_metrics.items():
        print(f"    {metric:12s}: {value:.4f}")
    
    print("\n   Métricas en TEST:")
    for metric, value in test_metrics.items():
        print(f"    {metric:12s}: {value:.4f}")
    
    print("\n   Matriz de Confusión (Test):")
    print(f"    TN: {cm[0,0]:4d}  FP: {cm[0,1]:4d}")
    print(f"    FN: {cm[1,0]:4d}  TP: {cm[1,1]:4d}")
    
    print(" Evaluación completada")
    
    return train_metrics, test_metrics, cm.tolist()

def save_model(model, output_path='models/model.joblib'):
    """Guarda el modelo entrenado"""
    print(f"\n Guardando modelo en: {output_path}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    
    print(" Modelo guardado exitosamente")

def save_metrics(train_metrics, test_metrics, confusion_matrix, output_path='metrics.json'):
    """Guarda las métricas en formato JSON"""
    print(f"\n Guardando métricas en: {output_path}")
    
    metrics = {
        'train': train_metrics,
        'test': test_metrics,
        'confusion_matrix': confusion_matrix
    }
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(" Métricas guardadas exitosamente")

def log_to_mlflow(model, model_params, train_metrics, test_metrics, model_path):
    """Registra experimento en MLflow"""
    if not USE_MLFLOW:
        return
    
    print("\n Registrando experimento en MLflow...")
    
    try:
        mlflow.set_experiment("telcovision-churn")
        
        with mlflow.start_run():
            # Log parámetros
            mlflow.log_params(model_params)
            
            # Log métricas de train
            for metric, value in train_metrics.items():
                mlflow.log_metric(f"train_{metric}", value)
            
            # Log métricas de test
            for metric, value in test_metrics.items():
                mlflow.log_metric(f"test_{metric}", value)
            
            # Log modelo
            mlflow.sklearn.log_model(model, "model")
            
            # Log artefacto del modelo local
            mlflow.log_artifact(model_path)
            
            print(" Experimento registrado en MLflow")
            
    except Exception as e:
        print(f"  Error al registrar en MLflow: {e}")

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Entrenamiento TelcoVision')
    parser.add_argument('--data', default='data/processed', help='Carpeta con datos procesados')
    parser.add_argument('--model', default='models/model.joblib', help='Ruta para guardar el modelo')
    parser.add_argument('--metrics', default='metrics.json', help='Ruta para guardar las métricas')
    parser.add_argument('--params', default='params.yaml', help='Archivo de parámetros')
    args = parser.parse_args()
    
    print("="*80)
    print(" ENTRENAMIENTO DEL MODELO - PROYECTO TELCOVISION")
    print("="*80)
    
    # 1. Cargar parámetros
    params = load_params(args.params)
    
    # 2. Cargar datos procesados
    X_train, X_test, y_train, y_test = load_processed_data(args.data)
    
    # 3. Entrenar modelo
    model, model_params = train_model(X_train, y_train, params)
    
    # 4. Evaluar modelo
    train_metrics, test_metrics, cm = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    # 5. Guardar modelo
    save_model(model, args.model)
    
    # 6. Guardar métricas
    save_metrics(train_metrics, test_metrics, cm, args.metrics)
    
    # 7. Log a MLflow (si está habilitado)
    log_to_mlflow(model, model_params, train_metrics, test_metrics, args.model)
    
    print("\n" + "="*80)
    print(" ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("="*80)
    print(f"\n Resultados finales:")
    print(f"  Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Test Precision: {test_metrics['precision']:.4f}")
    print(f"  Test Recall:    {test_metrics['recall']:.4f}")
    print(f"  Test F1-Score:  {test_metrics['f1_score']:.4f}")
    print(f"  Test ROC-AUC:   {test_metrics['roc_auc']:.4f}")

if __name__ == '__main__':
    main()
