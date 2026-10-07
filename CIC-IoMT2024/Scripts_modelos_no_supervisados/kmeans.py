import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, adjusted_rand_score, silhouette_score


def construccion_kmeans_pipeline(n_clusters, random_state=19, n_init=10):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=n_init
        ))
    ])

def seleccionar_mejor_k_por_silhouette(x_train, x_val, k_candidates=(2,3,4,5,6,8,10), random_state=19, n_init=10, sample_size=5000):
    mejor_k = None
    mejor_silhouette = -1

    for k in k_candidates:
        pipeline = construccion_kmeans_pipeline(n_clusters=k, random_state=random_state, n_init=n_init)
        pipeline.fit(x_train)
        clusters_val = pipeline.predict(x_val)

        if len(np.unique(clusters_val)) < 2:
            continue

        silhouette = silhouette_score(x_val, clusters_val, sample_size=sample_size, random_state=random_state)

        print(f"K={k} -> Silhouette={silhouette:.4f}")

        if silhouette > mejor_silhouette:
            mejor_silhouette = silhouette
            mejor_k = k

    if mejor_k is None:
        raise RuntimeError("No se pudo selecionar un valor valido de K.")
    
    return mejor_k, mejor_silhouette

def crear_mapeo_cluster_clase(clusters_val, y_val):
    mapping = {}
    y_val_array = np.array(y_val)

    for cluster in np.unique(clusters_val):
        indices = np.where(clusters_val == cluster)[0]
        clases, conteos = np.unique(y_val_array[indices], return_counts=True)

        clase_mayoritaria = clases[np.argmax(conteos)]
        mapping[int(cluster)] = clase_mayoritaria
    
    return mapping

def evaluar_kmeans_binario(pipeline, mapping, x_test, y_test, sample_size):
    clusters_test = pipeline.predict(x_test)
    predicciones = []

    for cluster in clusters_test:
        clase_predicha = mapping.get(int(cluster), 0)
        predicciones.append(clase_predicha)
    
    y_pred = np.array(predicciones)
    
    reporte_texto = classification_report(y_test, y_pred, labels=[0, 1], zero_division=0)
    reporte_dict = classification_report(y_test, y_pred, labels=[0, 1], output_dict=True, zero_division=0)
    matriz = confusion_matrix(y_test, y_pred, labels=[0, 1])
    ari = adjusted_rand_score(y_test, clusters_test)

    silhouette = np.nan
    if len(np.unique(clusters_test)) >= 2:
        silhouette =  silhouette_score(x_test, clusters_test, sample_size=sample_size, random_state=19)

    return reporte_texto, reporte_dict, matriz, ari, silhouette, mapping

def evaluar_kmeans_multiclase(pipeline, mapping, x_test, y_test, sample_size):
    clusters_test = pipeline.predict(x_test)
    predicciones = []

    for cluster in clusters_test:
        clase_predicha = mapping.get(int(cluster), "Benign")
        predicciones.append(clase_predicha)
    
    y_pred = np.array(predicciones)

    reporte_texto = classification_report(y_test, y_pred, zero_division=0)
    reporte_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    matriz = confusion_matrix(y_test, y_pred)
    ari = adjusted_rand_score(y_test, clusters_test)

    silhouette = np.nan
    if len(np.unique(clusters_test)) >= 2:
        silhouette =  silhouette_score(x_test, clusters_test, sample_size=sample_size, random_state=19)

    return reporte_texto, reporte_dict, matriz, ari, silhouette, mapping