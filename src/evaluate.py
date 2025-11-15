import pandas as pd
import numpy as np
import joblib
import json
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    roc_curve, 
    auc, 
    precision_recall_curve,
    classification_report
)
import os

def load_data_and_model(data_path, model_path):
    X_test = pd.read_csv(f"{data_path}/X_test.csv")
    y_test = pd.read_csv(f"{data_path}/y_test.csv").values.ravel()
    model = joblib.load(model_path)
    
    print(f"\nDatos cargados:")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_test: {y_test.shape}")
    print(f"  Modelo: {type(model).__name__}")
    
    return X_test, y_test, model

def plot_confusion_matrix(y_test, y_pred, output_path):
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    filepath = f"{output_path}/confusion_matrix.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nMatriz de confusion guardada: {filepath}")
    return cm

def plot_roc_curve(y_test, y_proba, output_path):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    filepath = f"{output_path}/roc_curve.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Curva ROC guardada: {filepath}")
    print(f"AUC: {roc_auc:.4f}")
    
    return roc_auc

def plot_precision_recall(y_test, y_proba, output_path):
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    filepath = f"{output_path}/precision_recall_curve.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Curva Precision-Recall guardada: {filepath}")
    print(f"PR-AUC: {pr_auc:.4f}")
    
    return pr_auc

def plot_feature_importance(model, X_test, output_path):
    if not hasattr(model, 'feature_importances_'):
        print("\nEl modelo no tiene feature importances")
        return None
    
    importances = model.feature_importances_
    feature_names = X_test.columns
    
    feat_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    plt.barh(feat_df['feature'], feat_df['importance'])
    plt.xlabel('Importance')
    plt.title('Top 10 Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    filepath = f"{output_path}/feature_importance.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Feature importance guardado: {filepath}")
    return feat_df.to_dict('records')

def save_classification_report(y_test, y_pred, output_path):
    report = classification_report(y_test, y_pred, 
                                   target_names=['No Churn', 'Churn'],
                                   output_dict=True)
    
    with open(f"{output_path}/classification_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    report_text = classification_report(y_test, y_pred, 
                                       target_names=['No Churn', 'Churn'])
    
    with open(f"{output_path}/classification_report.txt", 'w') as f:
        f.write("CLASSIFICATION REPORT\n")
        f.write("="*50 + "\n\n")
        f.write(report_text)
    
    print(f"\nReportes de clasificacion guardados en: {output_path}/")
    
    return report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    
    args = parser.parse_args()
    
    print("="*60)
    print("EVALUACION AVANZADA - TELCOVISION CHURN")
    print("="*60)
    
    os.makedirs(args.output, exist_ok=True)
    
    X_test, y_test, model = load_data_and_model(args.data, args.model)
    
    print("\nGenerando predicciones...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    cm = plot_confusion_matrix(y_test, y_pred, args.output)
    roc_auc = plot_roc_curve(y_test, y_proba, args.output)
    pr_auc = plot_precision_recall(y_test, y_proba, args.output)
    feat_imp = plot_feature_importance(model, X_test, args.output)
    
    report = save_classification_report(y_test, y_pred, args.output)
    
    metrics = {
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'confusion_matrix': cm.tolist(),
        'feature_importance': feat_imp
    }
    
    with open(f"{args.output}/advanced_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n" + "="*60)
    print("EVALUACION COMPLETADA")
    print("="*60)
    print(f"\nArchivos generados en: {args.output}/")
    print("  - confusion_matrix.png")
    print("  - roc_curve.png")
    print("  - precision_recall_curve.png")
    if feat_imp:
        print("  - feature_importance.png")
    print("  - classification_report.txt")
    print("  - classification_report.json")
    print("  - advanced_metrics.json\n")

if __name__ == '__main__':
    main()