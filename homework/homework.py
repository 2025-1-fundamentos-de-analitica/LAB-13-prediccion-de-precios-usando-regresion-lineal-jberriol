#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#

import pandas as pd
import gzip
import pickle
import json
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error
import os
from glob import glob 
# Cargar datos
def load_dataset():
    train_df = pd.read_csv("./files/input/train_data.csv.zip", index_col=False, compression="zip")
    test_df = pd.read_csv("./files/input/test_data.csv.zip", index_col=False, compression="zip")
    return train_df, test_df
# Separar características y respuesta
def separar_features_respuesta(df):
    return df.drop(columns=["Present_Price"]), df["Present_Price"]
# Preprocess
def preprocesamiento(df):
    df_copy = df.copy()
    reference_year = 2021
    df_copy["Age"] = reference_year - df_copy["Year"]
    df_copy = df_copy.drop(columns=['Year', 'Car_Name'])
    return df_copy
#Pipeline
def construir_pipeline(x_train):
    categorical_cols = ['Fuel_Type', 'Selling_type', 'Transmission']
    numerical_cols = [col for col in x_train.columns if col not in categorical_cols]

    data_preprocessor = ColumnTransformer(
        transformers=[
            ('categorical', OneHotEncoder(), categorical_cols),
            ('scaling', MinMaxScaler(), numerical_cols),
        ],
    )

    model_pipeline = Pipeline([
        ("preprocessing", data_preprocessor),
        ('feature_selection', SelectKBest(f_regression)),
        ('regressor', LinearRegression())
    ])
    return model_pipeline
#Estimacion
def configurar_estimador(pipeline):
    param_grid = {
        'feature_selection__k': range(1, 25),
        'regressor__fit_intercept': [True, False],
        'regressor__positive': [True, False]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        refit=True,
        verbose=1
    )

    return grid_search
# Crear directorio
def create_directory(directory_path):
    if os.path.exists(directory_path):
        for file in glob(f"{directory_path}/*"):
            os.remove(file)
        os.rmdir(directory_path)
    os.makedirs(directory_path)
#Guardar modelo
def save_trained_model(path, model):
    create_directory("files/models/")
    with gzip.open(path, "wb") as f:
        pickle.dump(model, f)
#Metricas
def compute_metrics(dataset_label, y_actual, y_predicted):
    return {
        "type": "metrics",
        "dataset": dataset_label,
        'r2': float(r2_score(y_actual, y_predicted)),
        'mse': float(mean_squared_error(y_actual, y_predicted)),
        'mad': float(median_absolute_error(y_actual, y_predicted)),
    }
# Flujo general
def start_pipeline():
    train_data, test_data = load_dataset()
    train_data = preprocesamiento(train_data)
    test_data = preprocesamiento(test_data)
    x_train, y_train = separar_features_respuesta(train_data)
    x_test, y_test = separar_features_respuesta(test_data)
    pipeline = construir_pipeline(x_train)

    model = configurar_estimador(pipeline)
    model.fit(x_train, y_train)

    save_trained_model("files/models/model.pkl.gz", model)

    y_test_predictions = model.predict(x_test)
    test_metrics = compute_metrics("test", y_test, y_test_predictions)
    
    y_train_predictions = model.predict(x_train)
    train_metrics = compute_metrics("train", y_train, y_train_predictions)

    os.makedirs("files/output/", exist_ok=True)
    with open("files/output/metrics.json", "w", encoding="utf-8") as file:
        file.write(json.dumps(train_metrics) + "\n")
        file.write(json.dumps(test_metrics) + "\n")


if __name__ == "__main__":
    start_pipeline()