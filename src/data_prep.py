# src/data_prep.py
"""
Script de preparación de datos para el proyecto TelcoVision
Realiza limpieza, transformación y división train/test
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import yaml
import os
import json
import argparse

def load_params(params_path='params.yaml'):
    """Carga los parámetros desde params.yaml"""
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)
    return params

def load_data(data_path):
    """Carga el dataset raw"""
    print(f" Cargando datos desde: {data_path}")
    df = pd.read_csv(data_path)
    print(f" Dataset cargado: {df.shape[0]} registros, {df.shape[1]} columnas")
    return df

def clean_data(df, params):
    """Limpia y prepara los datos"""
    print("\n Iniciando limpieza de datos...")
    
    # Crear copia
    df_clean = df.copy()
    
    # 1. Eliminar columna ID (no es feature)
    id_col = params['data_prep']['id_column']
    if id_col in df_clean.columns:
        df_clean = df_clean.drop(columns=[id_col])
        print(f"  ✓ Eliminada columna ID: {id_col}")
    
    # 2. Verificar valores nulos
    null_counts = df_clean.isnull().sum()
    if null_counts.sum() > 0:
        print(f"  ⚠ Valores nulos encontrados:\n{null_counts[null_counts > 0]}")
        df_clean = df_clean.dropna()
        print(f"  ✓ Filas con nulos eliminadas. Nuevas dimensiones: {df_clean.shape}")
    else:
        print("  ✓ No se encontraron valores nulos")
    
    # 3. Verificar duplicados
    duplicates = df_clean.duplicated().sum()
    if duplicates > 0:
        df_clean = df_clean.drop_duplicates()
        print(f"  ✓ Eliminados {duplicates} registros duplicados")
    else:
        print("  ✓ No se encontraron duplicados")
    
    print(f" Limpieza completada. Dimensiones finales: {df_clean.shape}")
    return df_clean

def encode_features(df, params):
    """Codifica variables categóricas"""
    print("\n Codificando variables categóricas...")
    
    df_encoded = df.copy()
    categorical_cols = params['data_prep']['categorical_features']
    
    encoders = {}
    
    for col in categorical_cols:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le
            print(f"  ✓ Codificada columna: {col} ({len(le.classes_)} categorías)")
    
    print(f" Codificación completada: {len(encoders)} columnas procesadas")
    return df_encoded, encoders

def split_data(df, params):
    """Divide el dataset en train y test"""
    print("\n  Dividiendo dataset en train/test...")
    
    target_col = params['data_prep']['target_column']
    test_size = params['data_prep']['test_size']
    random_state = params['data_prep']['random_state']
    
    # Separar features y target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Mantener proporción de churn
    )
    
    print(f"  ✓ Train set: {X_train.shape[0]} registros")
    print(f"  ✓ Test set: {X_test.shape[0]} registros")
    print(f"  ✓ Proporción de churn en train: {y_train.mean():.2%}")
    print(f"  ✓ Proporción de churn en test: {y_test.mean():.2%}")
    
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test, params):
    """Escala las variables numéricas"""
    print("\n📏 Escalando variables numéricas...")
    
    numeric_cols = params['data_prep']['numeric_features']
    
    # Copiar dataframes
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # Escalar solo numéricas
    scaler = StandardScaler()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    print(f"  ✓ Escaladas {len(numeric_cols)} columnas numéricas")
    print("Escalado completado")
    
    return X_train_scaled, X_test_scaled, scaler

def save_processed_data(X_train, X_test, y_train, y_test, output_dir='data/processed'):
    """Guarda los datos procesados"""
    print(f"\n Guardando datos procesados en: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar datasets
    X_train.to_csv(f'{output_dir}/X_train.csv', index=False)
    X_test.to_csv(f'{output_dir}/X_test.csv', index=False)
    y_train.to_csv(f'{output_dir}/y_train.csv', index=False, header=True)
    y_test.to_csv(f'{output_dir}/y_test.csv', index=False, header=True)
    
    print("  ✓ X_train.csv")
    print("  ✓ X_test.csv")
    print("  ✓ y_train.csv")
    print("  ✓ y_test.csv")
    print("Datos guardados exitosamente")

def save_metadata(encoders, scaler, params, output_dir='data/processed'):
    """Guarda metadatos del preprocesamiento"""
    print("\n📋 Guardando metadatos...")
    
    metadata = {
        'encoders': {col: enc.classes_.tolist() for col, enc in encoders.items()},
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'params': params['data_prep']
    }
    
    with open(f'{output_dir}/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("  ✓ metadata.json")
    print("Metadatos guardados")

def main():
    """Función principal"""
    # Parser de argumentos
    parser = argparse.ArgumentParser(description='Preparación de datos TelcoVision')
    parser.add_argument('--input', default='data/raw/telco_churn.csv', help='Ruta al dataset raw')
    parser.add_argument('--output', default='data/processed', help='Carpeta de salida')
    parser.add_argument('--params', default='params.yaml', help='Archivo de parámetros')
    args = parser.parse_args()
    
    print("="*80)
    print("PREPARACIÓN DE DATOS - PROYECTO TELCOVISION")
    print("="*80)
    
    # 1. Cargar parámetros
    params = load_params(args.params)
    
    # 2. Cargar datos raw
    df = load_data(args.input)
    
    # 3. Limpiar datos
    df_clean = clean_data(df, params)
    
    # 4. Codificar categóricas
    df_encoded, encoders = encode_features(df_clean, params)
    
    # 5. Dividir train/test
    X_train, X_test, y_train, y_test = split_data(df_encoded, params)
    
    # 6. Escalar features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, params)
    
    # 7. Guardar datos procesados
    save_processed_data(X_train_scaled, X_test_scaled, y_train, y_test, args.output)
    
    # 8. Guardar metadatos
    save_metadata(encoders, scaler, params, args.output)
    
    print("\n" + "="*80)
    print("PREPARACION DE DATOS COMPLETADA EXITOSAMENTE")
    print("="*80)

if __name__ == '__main__':
    main()
