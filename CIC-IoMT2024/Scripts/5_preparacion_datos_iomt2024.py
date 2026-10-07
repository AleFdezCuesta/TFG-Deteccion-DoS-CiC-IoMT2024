import os
import pandas as pd
from sklearn.model_selection import train_test_split

RUTA_TRAIN = "CIC-IoMT2024/Datos_procesados_iomt2024/train_preprocesado_con_etiquetas.csv"
RUTA_TEST = "CIC-IoMT2024/Datos_procesados_iomt2024/test_preprocesado_con_etiquetas.csv"
CARPETA_SALIDA = "CIC-IoMT2024/Datos_preparados_iomt2024"
os.makedirs(CARPETA_SALIDA, exist_ok=True)

VALIDATION_SIZE = 0.10
RANDOM_STATE = 19
STRATIFY = True

COLUMNAS_ETIQUETA = ["Label", "Attack_family", "Binary_label", "ZeroDay"]


def cargar_csv(ruta):
    df = pd.read_csv(ruta, low_memory=False)
    df.columns = df.columns.str.strip()
    return df

def normalizar_nombres_columnas(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.replace(" ", "_", regex=False)

    return df

def obtener_X_Y(df, tipo):
    X = df.drop(columns=COLUMNAS_ETIQUETA, errors="ignore")
    X = normalizar_nombres_columnas(X)
    
    if tipo == "binario":
        Y = df["Binary_label"].astype(int)
    elif tipo == "familia":
        Y = df["Attack_family"].astype(str)
    else:
        raise ValueError("Tipo debe ser 'binario' o 'familia'")
    
    return X, Y 

def dividir_train_y_validation(X, Y, val_size, random_state, stratify):
    if stratify:
        return train_test_split(X,Y, test_size = val_size, random_state=random_state,stratify=Y)
    else:
        return train_test_split(X, Y, test_size=val_size, random_state=random_state)
    
def guardar_x_y(X, Y, nombre_base):
    # Guardamos X e Y en archivos separados (mas limpio para los modelos)
    ruta_X = os.path.join(CARPETA_SALIDA, f"{nombre_base}_X.csv")
    ruta_Y = os.path.join(CARPETA_SALIDA, f"{nombre_base}_Y.csv")

    X.to_csv(ruta_X, index=False)
    Y.to_csv(ruta_Y, index=False)

    print(f"Guardado: {ruta_X}")
    print(f"Guardado: {ruta_Y}")

def preparar_y_guardar_por_tipo_objetivo(df_train_known, df_test_known, df_test_zeroday, tipo_objetivo, prefijo):
    X_train_all, Y_train_all = obtener_X_Y(df_train_known, tipo_objetivo)
    X_train, X_val, Y_train, Y_val = dividir_train_y_validation(X_train_all, Y_train_all, VALIDATION_SIZE, RANDOM_STATE, STRATIFY)

    print("\nDivision TRAIN/VAL (solo known):")
    print("X_train:", X_train.shape, ", Y_train: ",Y_train.shape)
    print("X_val: ", X_val.shape, ", Y_val: ", Y_val.shape)

    X_test_known, Y_test_known = obtener_X_Y(df_test_known, tipo_objetivo)
    print("\nTEST Known: X_test_known: ", X_test_known.shape, ", Y_test_known: ", Y_test_known.shape)

    X_test_zeroday = None
    Y_test_zeroday = None
    if df_test_zeroday is not None and len(df_test_zeroday) > 0:
        X_test_zeroday, Y_test_zeroday = obtener_X_Y(df_test_zeroday, tipo_objetivo)
        print("X_test_zeroday: ", X_test_zeroday.shape, ", y_test_zeroday: ", Y_test_zeroday.shape)
    else:
        print("\nNo hay muestras ZeroDay en TEST")

    guardar_x_y(X_train, Y_train, f"{prefijo}_train")
    guardar_x_y(X_val, Y_val, f"{prefijo}_val")
    guardar_x_y(X_test_known, Y_test_known, f"{prefijo}_test_known")

    if X_test_zeroday is not None:
        guardar_x_y(X_test_zeroday, Y_test_zeroday, f"{prefijo}_test_zeroday")

if __name__ == "__main__":
    print("SCRIPT DE PREPARACION DE DATOS - CiC - IoMT2024")
    print("Tamaño validacion (sobre TRAIN):", VALIDATION_SIZE * 100 ,"%")
    print("STRATIFY: ", STRATIFY)
    print("Salida: ", CARPETA_SALIDA)

    df_train = cargar_csv(RUTA_TRAIN)
    df_test = cargar_csv(RUTA_TEST)

    print("\nTamaño de los dataframes de test y train:")
    print("Train: ", df_train.shape)
    print("Test: ", df_test.shape)

    # Separamos TRAIN en Known vs ZeroDay (para que el entrenamiento no vea el ataque zero-day)
    df_train_known = df_train[df_train["ZeroDay"] == "Known"].copy()
    df_train_zeroday = df_train[df_train["ZeroDay"] == "ZeroDay"].copy()

    print("\nDistribucion ZeroDay en TRAIN (antes de separar):")
    print(df_train["ZeroDay"].value_counts())
    print("\nTamaños tras la separacion:")
    print("TRAIN Known muestras: ", df_train_known.shape)
    print(f"TRAIN ZeroDay muestras:  {df_train_zeroday.shape} Reservadas fuera del entrenamiento y validacion para mantener el escenario zero-day.")
    
    # Separamos TEST en Known vs ZeroDay (Evaluamos despues ambas)
    df_test_known = df_test[df_test["ZeroDay"] == "Known"].copy()
    df_test_zeroday = df_test[df_test["ZeroDay"] == "ZeroDay"].copy()

    print("\nDistribucion ZeroDay en TEST (antes de separar):")
    print(df_test["ZeroDay"].value_counts())
    print("Tamaños tras la separacion:")
    print("TEST Known muestras: ", df_test_known.shape)
    print("TEST ZeroDay muestras: ", df_test_zeroday.shape)

    preparar_y_guardar_por_tipo_objetivo(df_train_known, df_test_known, df_test_zeroday, "binario", "binario")
    preparar_y_guardar_por_tipo_objetivo(df_train_known, df_test_known, df_test_zeroday, "familia", "multiclase")

    print("\n Preparacion completada")