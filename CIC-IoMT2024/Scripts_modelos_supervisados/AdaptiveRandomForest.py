import os
import numpy as np
import matplotlib.pyplot as plt

from river import forest

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay

def construccion_arf_binario(n_models=10, max_features="sqrt", lambda_value=6, grace_period=50, split_criterion="info_gain", leaf_prediction="nba", seed=19):
    return forest.ARFClassifier(n_models=n_models, max_features=max_features, lambda_value=lambda_value, grace_period=grace_period, split_criterion=split_criterion, leaf_prediction=leaf_prediction, seed=seed)

def entrenar_arf_binario(modelo, x_train, y_train):
    for indice in range(len(x_train)):
        if hasattr(x_train, "iloc"):
            caracteristicas = x_train.iloc[indice].to_dict()
            etiqueta = int(y_train.iloc[indice])
        else:
            caracteristicas = dict(enumerate(x_train[indice]))
            etiqueta = int(y_train[indice])
        modelo.learn_one(caracteristicas, etiqueta)
    return modelo

def actualizar_arf_binario(modelo, x_nuevas_muestras, y_nuevas_muestras):
    for indice in range(len(x_nuevas_muestras)):
        if hasattr(x_nuevas_muestras, "iloc"):
            caracteristicas = x_nuevas_muestras.iloc[indice].to_dict()
            etiqueta = int(y_nuevas_muestras.iloc[indice])
        else:
            caracteristicas = dict(enumerate(x_nuevas_muestras[indice]))
            etiqueta = int(y_nuevas_muestras[indice])

        modelo.learn_one(caracteristicas, etiqueta)

    return modelo

def evaluar_arf_binario(modelo, x_test, y_test, carpeta_salida, nombre_grafica, threshold=0.5):
    y_prob = []
    for indice in range(len(x_test)):
        if hasattr(x_test, "iloc"):
            caracteristicas = x_test.iloc[indice].to_dict()
        else:
            caracteristicas = dict(enumerate(x_test[indice]))

        probabilidades = modelo.predict_proba_one(caracteristicas)
        prob_clase_1 = probabilidades.get(1, 0.0)

        y_prob.append(prob_clase_1)

    y_prob = np.array(y_prob)
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
        plt.title("Curva ROC - Adaptive Random Forest - Binario")
        plt.tight_layout()
        plt.savefig(os.path.join(carpeta_salida, nombre_grafica))
        plt.close()

    return reporte_texto, reporte_dict, matriz, auc
