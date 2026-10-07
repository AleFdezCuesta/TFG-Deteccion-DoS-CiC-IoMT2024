import os
import pandas as pd
import numpy as np

CARPETA_TRAIN = "CIC-IoMT2024/Dataset_Limpio/train"
CARPETA_TEST = "CIC-IoMT2024/Dataset_Limpio/test"

COLUMNAS_ANALISIS = ["Rate", "IAT", "Variance", "Tot sum", "Std"]

def listar_csv(carpeta):
    archivos = []
    for elem in os.listdir(carpeta):
        if elem.lower().endswith(".csv"):
            archivos.append(elem)
    archivos.sort()

    return archivos

def cargar_datos(carpeta):
    archivos = listar_csv(carpeta)
    lista_dfs = []

    print(f"Leyendo {len(archivos)} CSVs de {carpeta}\n")

    for nombre in archivos:
        ruta = os.path.join(carpeta, nombre)
        print(f"Leyendo {nombre}")

        df = pd.read_csv(
            ruta, low_memory=False, 
            na_values=["Infinity", "-Infinity", "inf", "-inf"]
        )

        df.columns = df.columns.str.strip()
        lista_dfs.append(df)

    df_total = pd.concat(lista_dfs, ignore_index=True)
    return df_total

def limpiar_infinitos_y_nan(df, carpeta):
    print(f"\nLimpieza de NaN e Inf en {carpeta}")
    filas_antes = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    filas_despues = len(df)
    print(f"Filas eliminadas por NaN/inf:{filas_antes - filas_despues}")
    return df

def reporte_percentiles_en_sigmas(df, columnas, percentil_inferior, percentil_superior):
    """
    Calculamos a cuantas desviaciones tipicas de la media estan los valores p1 y p99
    """
    print("ANALISIS DE PERCENTILES EN UNIDADES DE DESVIACION TIPICA")
    for col in columnas:
        mu = float(df[col].mean())
        sigma = float(df[col].std(ddof=1)) # ddof=0 para poblacion ; ddof=1 para muestral
        
        if sigma == 0.0: #Columna constante
            print(f"{col}: sigma=0 => Columna constante (no calculamos z-scores).")
            continue
        
        p1 = float(df[col].quantile(percentil_inferior))
        p99 = float(df[col].quantile(percentil_superior))

        z1 = (p1 - mu) / sigma
        z99 = (p99 - mu) / sigma

        print(f"{col}")
        print(f"Media: {mu:.6g}")
        print(f"Std: {sigma:.6g}")
        print(f"p1: {p1:.6g} (z={z1:.3f})")
        print(f"p99: {p99:.6g} (z={z99:.3f})\n")

if __name__ == "__main__":
    print("ANALISIS DE OUTLIERS POR DESVIACION TIPICA")
    print("Objetivo: Evaluar percentiles 1 y 99 en unidades de sigma\n")

    df_train = cargar_datos(CARPETA_TRAIN)
    df_test = cargar_datos(CARPETA_TEST)

    df_train_limpieza = limpiar_infinitos_y_nan(df_train, CARPETA_TRAIN)
    df_test_limpieza = limpiar_infinitos_y_nan(df_test, CARPETA_TEST)

    print("\n--- TRAIN ---")
    reporte_percentiles_en_sigmas(df_train_limpieza, COLUMNAS_ANALISIS, 0.01, 0.99)
    
    print("\n--- TEST ---")
    reporte_percentiles_en_sigmas(df_test_limpieza, COLUMNAS_ANALISIS, 0.01, 0.99)
