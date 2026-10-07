import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay

#Desbalanceo con SMOTE
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

def construccion_rf_pipeline(n_estimators, class_weight, n_jobs, random_state):
    # n_jobs = -1 para aprovechar todos los nucleos del procesador
    return Pipeline([("rf", RandomForestClassifier(n_estimators=n_estimators, random_state = random_state, n_jobs= n_jobs, class_weight=class_weight))])

def construccion_rf_pipeline_smote(n_estimators, class_weight, n_jobs, random_state):
    return Pipeline([("smote", SMOTE(random_state=random_state)), ("rf", RandomForestClassifier(n_estimators=n_estimators, random_state = random_state, n_jobs= n_jobs, class_weight=class_weight))])


def evaluar_rf_binario(pipeline, x_test, y_test, carpeta_salida, nombre_grafica, threshold=0.8):
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

def evaluar_rf_multiclase(pipeline, x_test, y_test):
    # Predicción
    y_pred = pipeline.predict(x_test)
   
    # Metricas principales
    reporte_texto = classification_report(y_test, y_pred, zero_division=0)
    reporte_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    matriz = confusion_matrix(y_test, y_pred)   

    auc = np.nan
    try:
        clases_modelo = pipeline.named_steps["rf"].classes_
        clases_presentes = np.unique(y_test)

        if len(clases_presentes) > 1 :
            y_prob = pipeline.predict_proba(x_test)
            auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro", labels=clases_modelo)
    except Exception:
        auc = np.nan

    return reporte_texto, reporte_dict, matriz, auc