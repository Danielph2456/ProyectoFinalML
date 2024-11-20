import kagglehub
import pandas as pd
from xgboost import XGBRegressor
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os

def download_and_load_data():
    """
    Descarga el dataset desde Kaggle y lo carga en un DataFrame.
    """
    # Descargar el dataset desde Kaggle
    path = kagglehub.dataset_download("shashanknecrothapa/ames-housing-dataset")
    print("Path to dataset files:", path)
    
    # Cargar el archivo CSV
    data_path = f"{path}/AmesHousing.csv"
    data = pd.read_csv(data_path)
    print("Datos cargados exitosamente.")
    return data

def preprocess_data(data):
    """
    Preprocesa los datos del dataset descargado.
    Valida la existencia de las columnas necesarias y selecciona las características correctas.
    """
    try:
        # Mostrar tamaño original del dataset
        print(f"Tamaño original del dataset: {data.shape}")
        
        # Seleccionar solo las columnas numéricas
        num_columns = data.select_dtypes(include=["number"]).columns
        print(f"Columnas numéricas: {list(num_columns)}")
        
        # Rellenar valores nulos solo en las columnas numéricas
        data[num_columns] = data[num_columns].fillna(data[num_columns].mean())
        print(f"Tamaño después de rellenar nulos: {data.shape}")
        
        # Verificar si la columna objetivo 'SalePrice' existe
        if "SalePrice" not in data.columns:
            print("Columnas disponibles en el dataset:")
            print(data.columns)
            raise ValueError("La columna 'SalePrice' no existe en el dataset. Verifica los datos descargados.")
        
        # Lista de características relevantes
        feature_columns = ['Overall Qual', 'Gr Liv Area', 'Garage Cars', 'Garage Area', 'Total Bsmt SF']
        
        # Verificar que todas las columnas requeridas existan en el dataset
        print("Verificando columnas requeridas...")
        for column in feature_columns:
            if column not in data.columns:
                print(f"Columnas disponibles en el dataset: {list(data.columns)}")
                raise ValueError(f"La columna requerida '{column}' no existe en el dataset.")
        
        # Seleccionar las características (X) y la variable objetivo (y)
        X = data[feature_columns]
        y = data["SalePrice"]
        
        # Verificar si quedan datos suficientes después del preprocesamiento
        if len(X) == 0 or len(y) == 0:
            raise ValueError("No hay suficientes datos después del preprocesamiento.")
        
        print(f"Tamaño final de las características (X): {X.shape}")
        print(f"Tamaño final de la variable objetivo (y): {y.shape}")
        return X, y
    except Exception as e:
        print(f"Error durante el preprocesamiento: {e}")
        raise



def train_and_save_model(X, y):
    """
    Entrena un modelo de XGBoost y lo guarda como un archivo .pkl.
    También guarda las columnas utilizadas en el entrenamiento.
    """
    # Verificar que haya suficientes datos para entrenamiento y prueba
    if len(X) < 2:
        raise ValueError("No hay suficientes datos para entrenamiento y prueba.")
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Crear y entrenar el modelo
    model = XGBRegressor()
    model.fit(X_train, y_train)
    
    # Evaluar el modelo
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE: {mse}, R2: {r2}")
    
    # Guardar el modelo entrenado
    model_path = "xgb_model.pkl"
    joblib.dump(model, model_path)
    print(f"Modelo guardado en {model_path}")
    
    # Guardar las columnas utilizadas en el entrenamiento
    columns_path = "model_columns.pkl"
    joblib.dump(X.columns.tolist(), columns_path)
    print(f"Columnas utilizadas guardadas en {columns_path}")

def load_or_train_model():
    """
    Carga un modelo previamente guardado o lo entrena si no existe.
    También carga las columnas utilizadas en el entrenamiento.
    """
    model_path = "xgb_model.pkl"
    columns_path = "model_columns.pkl"
    if os.path.exists(model_path) and os.path.exists(columns_path):
        # Cargar modelo existente y columnas
        model = joblib.load(model_path)
        columns = joblib.load(columns_path)
        print("Modelo y columnas cargados exitosamente.")
        return model, columns
    else:
        # Entrenar el modelo si no existe
        print("Modelo no encontrado. Descargando datos y entrenando modelo.")
        data = download_and_load_data()
        X, y = preprocess_data(data)
        train_and_save_model(X, y)
        model = joblib.load(model_path)
        columns = joblib.load(columns_path)
        return model, columns

def predict(data):
    """
    Realiza una predicción utilizando el modelo cargado.
    """
    model, columns = load_or_train_model()  # Asegura que el modelo y las columnas estén cargados o entrenados
    try:
        # Crear un DataFrame con las columnas esperadas
        input_data = pd.DataFrame([data], columns=columns)
        
        # Realizar la predicción
        prediction = model.predict(input_data)
        return prediction[0]  # Devuelve solo el valor de la predicción
    except Exception as e:
        raise ValueError(f"Error al realizar la predicción: {e}")

