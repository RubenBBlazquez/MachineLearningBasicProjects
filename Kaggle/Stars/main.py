import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.compose import ColumnTransformer, make_column_selector, make_column_transformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

if __name__ == '__main__':
    stars_df = read_csv('datasets/Stars.csv')

    # Número de filas, de columnas,
    # nombres de columnas, cuantos nulos hay por
    # columnas, qué tipos de datos hay.
    # Numero de filas por cada tipo de estrella.

    print('número Filas', stars_df.shape[0])
    print('número Columnas', stars_df.shape[1])
    print('Names Columns ', stars_df.columns)
    print('Info Estrellas ', stars_df.info())

    # eliminamos las filas que tengan 2 o más na
    stars_df.dropna(thresh=6, inplace=True)

    # pipeline para preprocesar las variables númericas con la media de los 3 vecinos
    # y el StandardScaler
    pipelineNumbers = make_pipeline(
        KNNImputer(n_neighbors=3),
        StandardScaler()
    )

    # pipeline para preprocesar las variables categoricas con el imputer y el oneHotEncoder
    pipelineCategorical = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder(),
    )

    # creamos el column transformer que aplicará el estandarizado de los valores números y de los categoricos
    preprocessing = ColumnTransformer([
        ('numerical', pipelineNumbers, make_column_selector(dtype_include=np.number)),
        ('categorical', pipelineCategorical, make_column_selector(dtype_include='object'))
    ], remainder="passthrough", sparse_threshold=0)

    # realizamos el preprocesado de la información
    df_preprocessed = preprocessing.fit_transform(stars_df)
    columns_df = list(map(lambda x: x.split('__')[1], preprocessing.get_feature_names_out()))
    df_preprocessed = pd.DataFrame(df_preprocessed, columns=columns_df)

    # establecemos una contaminación de 0.4 y predecimos con el isolation forest para sacar los outliers
    contamination = 0.4
    isolation_forest = IsolationForest(contamination=contamination)
    outliers = isolation_forest.fit_predict(df_preprocessed.values)

    # filtramos todos las predicciones y sacamos los outliers
    filter_outliers = list(filter(lambda x: x[1] == -1, enumerate(outliers)))

    # sacamos los outliers
    outliers = pd.DataFrame(map(lambda x: pd.Series(df_preprocessed.iloc[x[0]]), filter_outliers))
