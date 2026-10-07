import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
from xgboost import XGBClassifier

def construccion_xgb_pipeline(n_estimators = 400, max_depth = 8, learning_rate = 0.05, subsample = 0.9, colsample_bytree = 0.9, random_state = 19):
    return Pipeline([("scaler", StandardScaler()), ("xgb", XGBClassifier(n_estimators = n_estimators, max_depth = max_depth, learning_rate = learning_rate, subsample = subsample, colsample_bytree = colsample_bytree, eval_metric = "logloss", n_jobs = -1, random_state = random_state, use_label_encoder = False))])

def evaluar_xgb_binario(pipeline, x_test, y_test, carpeta_salida, nombre_grafica, threshold=0.8):
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
        plt.title("Curva ROC - XGBoost - Binario)")
        plt.tight_layout()
        plt.savefig(f"{carpeta_salida}/{nombre_grafica}")
        plt.close()

    return reporte_texto, reporte_dict, matriz, auc

def evaluar_xgb_multiclase(pipeline, x_test, y_test):
    # Predicción
    y_pred = pipeline.predict(x_test)
   
    # Metricas principales
    reporte_texto = classification_report(y_test, y_pred, zero_division=0)
    reporte_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    matriz = confusion_matrix(y_test, y_pred)   

    return reporte_texto, reporte_dict, matriz
