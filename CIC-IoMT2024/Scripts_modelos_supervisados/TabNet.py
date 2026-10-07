import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay

from pytorch_tabnet.tab_model import TabNetClassifier
import torch

def construccion_tabnet_pipeline(n_d = 8, n_a = 8, n_steps = 3, gamma = 1.3, lambda_sparse = 1e-3, seed = 19, lr = 2e-2, step_size = 50, step_gamma = 0.9):
    tabnet = TabNetClassifier(n_d = n_d, n_a = n_a, n_steps = n_steps, gamma = gamma, lambda_sparse = lambda_sparse, seed = seed, optimizer_fn = torch.optim.Adam, optimizer_params = dict(lr = lr), scheduler_fn = torch.optim.lr_scheduler.StepLR, scheduler_params = {"step_size": step_size, "gamma": step_gamma}, verbose = 0)

    return Pipeline([("scaler", StandardScaler()), ("tabnet", tabnet)])

def entrenar_tabnet_pipeline(pipeline, x_train, y_train, x_val=None, y_val=None, max_epochs=60, patience=10, batch_size=2048, virtual_batch_size=256):
    scaler = pipeline.named_steps["scaler"]
    tabnet = pipeline.named_steps["tabnet"]

    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    
    y_train_np = np.asarray(y_train)
    y_val_np = np.asarray(y_val)

    tabnet.fit(x_train_scaled, y_train_np, eval_set=[(x_val_scaled, y_val_np)], max_epochs=max_epochs, patience=patience, batch_size=batch_size, virtual_batch_size=virtual_batch_size)

    return pipeline

def evaluar_tabnet_binario(pipeline, x_test, y_test, carpeta_salida, nombre_grafica, threshold=0.8):
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
        RocCurveDisplay.from_predictions(y_test, y_prob)
        plt.title("Curva ROC - TabNet - Binario")
        plt.tight_layout()
        plt.savefig(f"{carpeta_salida}/{nombre_grafica}")
        plt.close()

    return reporte_texto, reporte_dict, matriz, auc

def evaluar_tabnet_multiclase(pipeline, x_test, y_test):
    # Predicción
    y_pred = pipeline.predict(x_test)
   
    # Metricas principales
    reporte_texto = classification_report(y_test, y_pred, zero_division=0)
    reporte_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    matriz = confusion_matrix(y_test, y_pred)   

    return reporte_texto, reporte_dict, matriz