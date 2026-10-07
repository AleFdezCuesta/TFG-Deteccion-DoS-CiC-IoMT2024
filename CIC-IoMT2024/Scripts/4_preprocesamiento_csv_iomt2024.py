import os
import re
import pandas as pd
import numpy as np

CARPETA_TRAIN = "CIC-IoMT2024/Dataset_Limpio/train"
CARPETA_TEST = "CIC-IoMT2024/Dataset_Limpio/test"

CARPETA_SALIDA = "CIC-IoMT2024/Datos_procesados_iomt2024"
os.makedirs(CARPETA_SALIDA, exist_ok=True)

#Columnas a eliminar:
# - "Protocol Type" es una variable categorica codificada como entero (valores como: 1,3,6,13)
# ademas es redundante con las columnas binarias TCP/UDP/ICMP/... por lo que se elimina
# para evitar interpretaciones ordinales y redundancia.
COLUMNAS_A_ELIMINAR = ["Protocol Type"]
INCLUIR_COLUMNAS_A_ELIMINAR = False # En algunas pruebas finales de modelos querremos comprobar como afecta el dejar esta columna

ZERO_DAY_ATTACK = "TCP_IP-DoS-ICMP"
COLUMNAS_ETIQUETA = ["Label", "Attack_family", "Binary_label", "ZeroDay"]

K_SIGMAS = 4.0

def listar_csv(carpeta):
    archivos = []
    for elem in os.listdir(carpeta):
        if elem.lower().endswith(".csv"):
            archivos.append(elem)
    archivos.sort()

    return archivos

def nombre_label(nombre_archivo):
    base = os.path.basename(nombre_archivo)
    label = re.sub(r"(_train|_test)\.csv$", "", base, flags=re.IGNORECASE)
    return label
    
def mapear_attack_family(label):
    etiqueta = label.upper()

    if etiqueta == "BENIGN":
        return "Benign"
    if "DDOS" in etiqueta:
        return "DDoS"
    if "DOS" in etiqueta:
        return "DoS"
    if etiqueta.startswith("RECON-"):
        return "Recon"
    if etiqueta.startswith("ARP_"):
        return "Spoofing"
    if "MALFORMED" in etiqueta:
        return "Malformed"

    return "Other"
    
def grupo_zero_day(label):
    etiqueta = label.upper()
    if etiqueta == ZERO_DAY_ATTACK.upper():
        return "ZeroDay"
    return "Known"
    
def cargar_y_concatenar_etiquetando(carpeta):
    archivos = listar_csv(carpeta)
    if len(archivos) == 0:
        print(f"No hay CSV que leer en: {carpeta}")
        return pd.DataFrame()
    
    lista_dfs = []
    print(f"\nLeyendo {len(archivos)} CSVs de: {carpeta}\n")
    
    for i, nombre in enumerate(archivos, start=1):
        ruta = os.path.join(carpeta, nombre)
        print(f"[{i}/{len(archivos)}] {nombre}")

        df = pd.read_csv(
            ruta, low_memory=False, 
            na_values=["Infinity", "-Infinity", "inf", "-inf"]
        )

        df.columns = df.columns.str.strip()

        label = nombre_label(nombre)
        familia = mapear_attack_family(label)
        zero_day = grupo_zero_day(label)

        df["Label"] = label
        df["Attack_family"] = familia
        if familia == "Benign":
            df["Binary_label"] = 0
        else:
            df["Binary_label"] = 1
        df["ZeroDay"] = zero_day

        if not INCLUIR_COLUMNAS_A_ELIMINAR:
            df = df.drop(columns=COLUMNAS_A_ELIMINAR, errors="ignore")

        lista_dfs.append(df)

    df_total = pd.concat(lista_dfs, ignore_index=True)
    return df_total 

def limpiar_infinitos_y_nan(df):
    filas_antes = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    filas_despues = len(df)
    print(f"Filas eliminadas por NaN/inf:{filas_antes - filas_despues}")
    return df

def aplicar_capping_por_std(df, k):
    columnas_numericas = df.select_dtypes(include=[np.number]).columns
    columnas_excluir = ["Binary_label"]

    for col in columnas_numericas:
        if col in columnas_excluir:
            continue

        mu = df[col].mean()
        sigma = df[col].std(ddof=1)

        if sigma == 0 or pd.isna(sigma):
            continue
        
        limite_inferior = mu - k * sigma
        limite_superior = mu + k * sigma

        df[col] = df[col].clip(limite_inferior, limite_superior)
    return df


if __name__ == "__main__":

    print("SCRIPT DE PREPROCESAMIENTO - CIC-IoMT2024 - Dataset_Limpio")
    print("\nObjetivos:")
    print("- Cargar CSVs concatenados de train/test")
    print("- Añadir columnas de etiqueta (Label, Attack_family, Binary_label, ZeroDay)")
    print("- Limpiar NaN/inf y aplicar capping\n")
    
    print("Zero-day elegido:", ZERO_DAY_ATTACK)
    print("Columnas a eliminar: ", COLUMNAS_A_ELIMINAR)
    
    df_train = cargar_y_concatenar_etiquetando(CARPETA_TRAIN)
    df_test = cargar_y_concatenar_etiquetando(CARPETA_TEST)

    print("\nTAMAÑOS INICIALES DE LOS DATAFRAMES CONCATENADOS DE TEST Y TRAIN ANTES DE LIMPIEZA (con etiquetas añadidas)")
    print("DATAFRAME TRAIN: ", df_train.shape)
    print("DATAFRAME TEST: ", df_test.shape)

    print("\nDISTRIBUCION DE TRAIN Y TEST ANTES DE LIMPIEZA DE DATOS")
    print("\nDISTRIBUCION TRAIN (Attack_family):",df_train["Attack_family"].value_counts())
    print("\nDISTRIBUCION TRAIN (ZeroDay):", df_train["ZeroDay"].value_counts())
    print("\nDISTRIBUCION TRAIN (Label):", df_train["Label"].value_counts())

    print("\nDISTRIBUCION TEST (Attack_family):", df_test["Attack_family"].value_counts())
    print("\nDISTRIBUCION TEST (ZeroDay):", df_test["ZeroDay"].value_counts())
    print("\nDISTRIBUCION TEST (Label):", df_test["Label"].value_counts())

    print("\nLIMPIEZA DE LOS DATAFRAMES DE TEST Y TRAIN (Eliminamos NaN e inf)")
    print("TRAIN:")
    df_train_limpio = limpiar_infinitos_y_nan(df_train)
    print("TEST:")
    df_test_limpio = limpiar_infinitos_y_nan(df_test)
    
    print("\nTAMAÑOS FINALES DE LOS DATAFRAMES CONCATENADOS DE TEST Y TRAIN DESPUES DE LIMPIEZA")
    print("TRAIN: ", df_train_limpio.shape)
    print("TEST: ", df_test_limpio.shape)

    print("\nDISTRIBUCION DE TRAIN Y TEST DESPUES DE LIMPIEZA DE DATOS")
    print("\nDISTRIBUCION TRAIN (Attack_family):",df_train_limpio["Attack_family"].value_counts())
    print("\nDISTRIBUCION TRAIN (ZeroDay):", df_train_limpio["ZeroDay"].value_counts())
    print("\nDISTRIBUCION TRAIN (Label):", df_train_limpio["Label"].value_counts())

    print("\nDISTRIBUCION TEST (Attack_family):", df_test_limpio["Attack_family"].value_counts())
    print("\nDISTRIBUCION TEST (ZeroDay):", df_test_limpio["ZeroDay"].value_counts())
    print("\nDISTRIBUCION TEST (Label):", df_test_limpio["Label"].value_counts())

    print("\nAPLICAMOS CAPPING POR DESVIACION TIPICA")
    df_train_capping = aplicar_capping_por_std(df_train_limpio, K_SIGMAS)
    df_test_capping = aplicar_capping_por_std(df_test_limpio, K_SIGMAS)
    print("Capping aplicado a los dataframes de train y test\n")

    df_train_sin_etiquetas = df_train_capping.drop(columns=COLUMNAS_ETIQUETA, errors= "ignore")
    df_test_sin_etiquetas = df_test_capping.drop(columns=COLUMNAS_ETIQUETA, errors= "ignore")

    df_train_labels = df_train_capping[COLUMNAS_ETIQUETA].copy()
    df_test_labels = df_test_capping[COLUMNAS_ETIQUETA].copy()

    ruta_salida_train_con_etiquetas = os.path.join(CARPETA_SALIDA, "train_preprocesado_con_etiquetas.csv")
    ruta_salida_test_con_etiquetas = os.path.join(CARPETA_SALIDA, "test_preprocesado_con_etiquetas.csv")

    ruta_salida_train_sin_etiquetas = os.path.join(CARPETA_SALIDA, "train_preprocesado_sin_etiquetas.csv")
    ruta_salida_test_sin_etiquetas = os.path.join(CARPETA_SALIDA, "test_preprocesado_sin_etiquetas.csv")

    ruta_salida_train_solo_etiquetas = os.path.join(CARPETA_SALIDA, "train_solo_etiquetas.csv")
    ruta_salida_test_solo_etiquetas = os.path.join(CARPETA_SALIDA, "test_solo_etiquetas.csv")

    print(f"\n GUARDAMOS LOS DATAFRAMES DE TEST Y TRAIN EN LA CARPETA '{CARPETA_SALIDA}'")
    print("Archivo Train preprocesado con etiquetas: ", ruta_salida_train_con_etiquetas)
    print("Archivo Test preprocesado con etiquetas: ", ruta_salida_test_con_etiquetas)
    print("Archivo Train preprocesado sin etiquetas: ", ruta_salida_train_sin_etiquetas)
    print("Archivo Test preprocesado sin etiquetas: ", ruta_salida_test_sin_etiquetas)
    print("Archivo Train preprocesado SOLO etiquetas: ", ruta_salida_train_solo_etiquetas)
    print("Archivo Test preprocesado SOLO etiquetas: ", ruta_salida_test_solo_etiquetas)

    df_train_capping.to_csv(ruta_salida_train_con_etiquetas, index=False)
    df_test_capping.to_csv(ruta_salida_test_con_etiquetas, index=False)

    df_train_sin_etiquetas.to_csv(ruta_salida_train_sin_etiquetas, index=False)
    df_test_sin_etiquetas.to_csv(ruta_salida_test_sin_etiquetas, index=False)

    df_train_labels.to_csv(ruta_salida_train_solo_etiquetas, index=False)
    df_test_labels.to_csv(ruta_salida_test_solo_etiquetas, index=False)


    print("\nGuardado completado")
