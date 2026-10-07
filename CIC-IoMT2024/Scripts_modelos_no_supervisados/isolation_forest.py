import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay

def construccion_isolation_pipeline(n_estimators=200, max_samples="auto", contamination="auto", random_state=19, n_jobs=-1):
    return Pipeline([("scaler", StandardScaler()), ("isolation", IsolationForest(n_estimators=n_estimators, max_samples=max_samples, contamination=contamination, random_state=random_state, n_jobs=n_jobs))])

def seleccionar_umbral_por_f1(pipeline, x_val, y_val, percentiles=range(1, 21)):
    scores_val = pipeline.score_samples(x_val)
    mejor_f1 = -1
    mejor_umbral = None
    mejor_percentil = None

    for percentil in percentiles:
        umbral = np.percentile(scores_val, percentil)
        predicciones = []
        for score in scores_val:
            if score < umbral:
                predicciones.append(1)
            else:
                predicciones.append(0)
        y_pred_val = np.array(predicciones)

        tp = np.sum((y_pred_val == 1) & (y_val == 1))
        fp = np.sum((y_pred_val == 1) & (y_val == 0))
        fn = np.sum((y_pred_val == 0) & (y_val == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        if f1 > mejor_f1:
            mejor_f1 = f1
            mejor_umbral = umbral
            mejor_percentil = percentil
        
    return mejor_umbral, mejor_f1, mejor_percentil
    
def evaluar_isolation_binario(pipeline, x_test, y_test, umbral, carpeta_salida, nombre_grafica):
    scores_test = pipeline.score_samples(x_test)
    predicciones = []

    for score in scores_test:
        if score < umbral:
            predicciones.append(1)
        else:
            predicciones.append(0)
    y_pred = np.array(predicciones)

    reporte_texto = classification_report(y_test, y_pred, labels=[0, 1], zero_division=0)
    reporte_dict = classification_report(y_test, y_pred, labels=[0, 1], output_dict=True, zero_division=0)
    matriz = confusion_matrix(y_test, y_pred, labels=[0, 1])
    auc = np.nan

    if len(np.unique(y_test)) == 2:
        try:
            auc = roc_auc_score(y_test, -scores_test)
       
            os.makedirs(carpeta_salida, exist_ok=True)
            plt.figure()
            RocCurveDisplay.from_predictions(y_test, -scores_test)
            plt.title("Curva ROC - Isolation Forest - Binario)")
            plt.tight_layout()
            plt.savefig(f"{carpeta_salida}/{nombre_grafica}")
            plt.close()
        except Exception:
            auc = np.nan
    
    return reporte_texto, reporte_dict, matriz, auc
    