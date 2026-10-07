import os
import re
import pandas as pd

CARPETA_TRAIN = "CIC-IoMT2024/DataSetCICIoMT2024/train"
CARPETA_TEST = "CIC-IoMT2024/DataSetCICIoMT2024/test"

CARPETA_SALIDA = "CIC-IoMT2024/Dataset_Limpio"

CARPETA_SALIDA_TRAIN = os.path.join(CARPETA_SALIDA, "train")
CARPETA_SALIDA_TEST = os.path.join(CARPETA_SALIDA, "test")

def crear_carpetas_salida():
    os.makedirs(CARPETA_SALIDA_TRAIN, exist_ok=True)
    os.makedirs(CARPETA_SALIDA_TEST, exist_ok=True)

def limpiar_familia(nombre_archivo):
    base = nombre_archivo

    if base.lower().endswith(".pcap.csv"):
        base = base[:-len(".pcap.csv")]

    base = re.sub(r"_(train|test)$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"\d+$", "", base)

    return base

def agrupar_por_familia(carpeta_origen):

    grupos = {}

    for nombre in sorted(os.listdir(carpeta_origen)):
        
        familia = limpiar_familia(nombre)
        ruta = os.path.join(carpeta_origen, nombre)

        if familia not in grupos:
            grupos[familia] = []
        grupos[familia].append(ruta)

    return grupos

def concatenar_y_guardar(grupos, carpetas_salida, split):
    for familia, rutas in grupos.items():
        print(f"[{split}] Unificando {familia} (Total Archivos:{len(rutas)})")

        dataframes = []
        for ruta in rutas:
            df = pd.read_csv(ruta)
            dataframes.append(df)

        df_final = pd.concat(dataframes, ignore_index=True)

        nombre_salida = f"{familia}_{split}.csv"
        ruta_salida = os.path.join(carpetas_salida, nombre_salida)
        df_final.to_csv(ruta_salida, index=False)

        print(f"Guardado en {ruta_salida} (Filas: {df_final.shape[0]}, Columnas: {df_final.shape[1]})")


if __name__ == "__main__":
    crear_carpetas_salida()

    grupos_train = agrupar_por_familia(CARPETA_TRAIN)
    concatenar_y_guardar(grupos_train, CARPETA_SALIDA_TRAIN, "train")

    grupos_test = agrupar_por_familia(CARPETA_TEST)
    concatenar_y_guardar(grupos_test, CARPETA_SALIDA_TEST, "test")
