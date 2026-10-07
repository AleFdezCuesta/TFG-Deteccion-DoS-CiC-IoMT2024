import os
import pandas as pd
import numpy as np

CARPETA_TRAIN = "CIC-IoMT2024/Dataset_Limpio/train"
CARPETA_TEST = "CIC-IoMT2024/Dataset_Limpio/test"

def listar_csv(carpeta_datos):
    archivos = []
    for elemento in os.listdir(carpeta_datos):
        if elemento.lower().endswith(".csv"):
            archivos.append(elemento)
    archivos.sort()
    return archivos

def analizar_carpeta(carpeta_datos, nombre):
    """
    Analizamos todos los CSV de una carpeta (train o test)
    - Imprimimos nº de muestras y nº de columnas por archivo (familia)
    - Comprueba columnas y tipos
    - Calculamos NaN e Inf por archivo: Mirando cuantas filas tienen NaN/inf,
      si estan concentrados en una fila o columna concreta
    - Mostramos un resumen 
    """
    
    archivos = listar_csv(carpeta_datos)

    print(f"\nANALISIS DE {nombre.upper()}")
    print(f"Ruta de la carpeta: {carpeta_datos}")
    print(f"Total archivos (familias): {len(archivos)}\n")
    
    resumen = []
    columnas_dataset = None

    for archivo in archivos:
        ruta_archivo = os.path.join(carpeta_datos, archivo)
        df = pd.read_csv(ruta_archivo)
        df.columns = df.columns.str.strip()

        if columnas_dataset is None:
            columnas_dataset = list(df.columns)
            
        #NaN (Nulos)
        total_nan = int(df.isna().sum().sum())
        filas_con_nan = int(df.isna().any(axis=1).sum())
        nan_por_fila = df.isna().sum(axis=1)
        filas_con_2_o_mas_nan= int((nan_por_fila >= 2).sum())

        nan_por_columna = df.isna().sum()
        columnas_con_nan = nan_por_columna[nan_por_columna > 0].sort_values(ascending=False)
        columna_con_mas_nan = nan_por_columna.idxmax()
        nan_max = int(nan_por_columna.max())

        #INFINITOS
        df_num = df.select_dtypes(include=[np.number])
        total_inf = int(np.isinf(df_num.to_numpy()).sum())
        filas_con_inf = int(np.isinf(df_num.to_numpy()).any(axis=1).sum())
        inf_por_fila= np.isinf(df_num.to_numpy()).sum(axis=1)
        fila_con_2_o_mas_inf= int((inf_por_fila >= 2).sum())

        inf_por_columna= pd.Series(np.isinf(df_num.to_numpy()).sum(axis=0), index=df_num.columns)
        columnas_con_inf = inf_por_columna[inf_por_columna > 0].sort_values(ascending=False)
        columna_con_mas_inf =inf_por_columna.idxmax()
        inf_max = int(inf_por_columna.max())


        #Resumen
        print("=========================================================================")
        print(f"Archivo: {archivo}")
        print(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
        print(f"NaN totales: {total_nan} | Filas con NaN: {filas_con_nan} | Filas con 2 o mas NaN: {filas_con_2_o_mas_nan}")
        print(f"Inf totales: {total_inf} | Filas con Inf: {filas_con_inf} | Filas con 2 o mas Inf: {fila_con_2_o_mas_inf}")
        if total_nan == 0:
            print("Columnas con NaN: Ninguna")
        else:
            print(f"Columnas con valores NaN: {columnas_con_nan}")
        if total_inf == 0:
            print("Columnas con valores infinitos: Ninguna")
        else:
            print(f"Columnas con valores infinitos: {columnas_con_inf}")
        print(f"Columna con mas NaN: {columna_con_mas_nan} con {nan_max} valores NaN")
        print(f"Columna con mas Inf: {columna_con_mas_inf} con {inf_max} valores Inf")
        print("=========================================================================\n")
       
        resumen.append({
            "archivo": archivo,
            "filas": df.shape[0],
            "columnas": df.shape[1],
            "total_nan": total_nan,
            "filas_con_nan": filas_con_nan,
            "total_inf": total_inf,
            "filas_con_inf": filas_con_inf,
        })

        columnas_dataset = list(df.columns)

    df_resumen = pd.DataFrame(resumen)

    print(f"----- RESUMEN GLOBAL ({nombre.upper()}) -----")
    print("Total Filas:", int(df_resumen["filas"].sum()))
    print("Total Columnas (por archivo):", int(df_resumen["columnas"].iloc[0]))
    print("Total NaN:", int(df_resumen["total_nan"].sum()))
    print("Total Inf:", int(df_resumen["total_inf"].sum()))
    print("Total de filas con NaN:", int(df_resumen["filas_con_nan"].sum()))
    print("Total filas con Inf:", int(df_resumen["filas_con_inf"].sum()))
    print("Listado de columnas:", columnas_dataset)
   
    return df_resumen

if __name__ == "__main__":
    
    df_test = analizar_carpeta(CARPETA_TEST, "test")
    df_train = analizar_carpeta(CARPETA_TRAIN, "train")
    
    print("----- RESUMEN FINAL TAMAÑO CARPETAS TRAIN Y TEST -----\n")
    print(f"TEST: Familias: {len(df_test)}, Filas totales: {int(df_test['filas'].sum())}")
    print(f"TRAIN: Familias: {len(df_train)}, Filas totales: {int(df_train['filas'].sum())}")
   


