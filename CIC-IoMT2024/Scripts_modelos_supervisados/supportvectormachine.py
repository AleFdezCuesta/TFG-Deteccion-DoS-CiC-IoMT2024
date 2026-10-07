import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay

def construccion_svm_pipeline_binario(kernel = 'rbf', C = 1.0, gamma = 'scale', random_state = 19, class_weight="balanced"):

    return Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel=kernel, C=C, gamma = gamma, probability=True, random_state=random_state, class_weight=class_weight))
    ])

def construccion_svm_pipeline_multiclase(C = 1.0, random_state=19, class_weight="balanced", max_iter=20000):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svm', LinearSVC(C=C, random_state=random_state, class_weight=class_weight, max_iter= max_iter))
    ])

def evaluar_svm_binario(pipeline, x_test, y_test, carpeta_salida, nombre_grafica, threshold=0.8):
   # y_pred = pipeline.predict(x_test)
    y_prob = pipeline.predict_proba(x_test)[:,1]
    y_pred = (y_prob > threshold).astype(int)

    clases_presentes = np.unique(y_test)
    
    reporte_texto = classification_report(y_test, y_pred, labels=[0, 1], zero_division=0) 
    reporte_dict = classification_report(y_test, y_pred, labels=[0, 1], output_dict=True, zero_division=0) 
    matriz = confusion_matrix(y_test, y_pred, labels=[0, 1]) 
    
    auc = np.nan

    if len(clases_presentes) == 2:
        try:
            auc = roc_auc_score(y_test, y_prob)
        except Exception:
            auc = np.nan

        os.makedirs(carpeta_salida, exist_ok=True)
        plt.figure()
        RocCurveDisplay.from_estimator(pipeline, x_test, y_test)
        plt.title("Curva ROC - Support Vector Machine - Binario)")
        plt.tight_layout()
        plt.savefig(f"{carpeta_salida}/{nombre_grafica}")
        plt.close()

    return reporte_texto, reporte_dict, matriz, auc


def evaluar_svm_multiclase(pipeline, x_test, y_test):
    # Predicción
    y_pred = pipeline.predict(x_test)
   
    # Metricas principales
    reporte_texto = classification_report(y_test, y_pred, zero_division=0)
    reporte_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    matriz = confusion_matrix(y_test, y_pred)   

    return reporte_texto, reporte_dict, matriz