import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
from lightgbm import LGBMClassifier

def construccion_lgbm_pipeline_binario(objective="binary", n_estimators = 400, num_leaves = 32, learning_rate = 0.05, subsample = 0.9, colsample_bytree = 0.9,random_state = 19, n_jobs=-1, verbose=-1):
    return Pipeline([("lgbm", LGBMClassifier(objective=objective, n_estimators = n_estimators, num_leaves = num_leaves, learning_rate = learning_rate, subsample = subsample, colsample_bytree = colsample_bytree, random_state = random_state, n_jobs = n_jobs, verbose=verbose))])

def construccion_lgbm_pipeline_multiclase(objective="multiclass", n_estimators = 400, num_leaves = 32, learning_rate = 0.05, subsample = 0.9, colsample_bytree = 0.9,random_state = 19, n_jobs=-1, verbose=-1):
    return Pipeline([("lgbm", LGBMClassifier(objective=objective, n_estimators = n_estimators, num_leaves = num_leaves, learning_rate = learning_rate, subsample = subsample, colsample_bytree = colsample_bytree, random_state = random_state, n_jobs = n_jobs, verbose=verbose))])

def evaluar_lgbm_binario(pipeline, x_test, y_test, carpeta_salida, nombre_grafica, threshold=0.8):
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
        plt.title("Curva ROC - Random Forest - Binario)")
        plt.tight_layout()
        plt.savefig(f"{carpeta_salida}/{nombre_grafica}")
        plt.close()

    return reporte_texto, reporte_dict, matriz, auc

def evaluar_lgbm_multiclase(pipeline, x_test, y_test):
    # Predicción
    y_pred = pipeline.predict(x_test)
   
    # Metricas principales
    reporte_texto = classification_report(y_test, y_pred, zero_division=0)
    reporte_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    matriz = confusion_matrix(y_test, y_pred)   

    auc = np.nan
    try:
        clases_modelo = pipeline.named_steps["lgbm"].classes_
        clases_presentes = np.unique(y_test)

        if len(clases_presentes) > 1 :
            y_prob = pipeline.predict_proba(x_test)
            auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro", labels=clases_modelo)
    except Exception:
        auc = np.nan

    return reporte_texto, reporte_dict, matriz, auc