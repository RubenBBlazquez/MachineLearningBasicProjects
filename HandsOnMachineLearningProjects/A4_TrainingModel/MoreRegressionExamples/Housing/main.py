import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

if __name__ == '__main__':
    housing = pd.read_csv('casas.csv')

    # 1.	Explorar los datos.
    # Ver número de filas y de columnas.
    print('filas, ', housing.shape[0], ', columnas: ', housing.shape[1])

    # Comprobar los tipos de datos de las columnas
    print(housing.dtypes.values)

    # Comprobar las que son categóricas y las que no
    cat_columns = filter(lambda x: x[1] == object, enumerate(housing.dtypes))
    cat_columns = list(map(lambda x: housing.columns[x[0]], cat_columns))
    print(cat_columns)

    num_columns = list(filter(lambda x: x not in cat_columns, housing.columns))
    print(num_columns)

    # Comprobar los valores distintos que se repiten de cada característica
    print(housing.nunique())

    # Comprobar las estadísticas de las columnas
    print(housing[num_columns].describe())
    print(housing[cat_columns].describe())

    # 2. Representar el diagrama de distribución del precio de la casa, para ver si es una distribución normal.
    plt.figure(figsize=(20, 10))
    housing[num_columns].hist()
    plt.show()

    # Comprobar gráfica de valores outliers, en este caso para el área. Las otras variables numéricas son discretas.
    plt.boxplot(housing['area'])
    plt.show()

    plt.boxplot(housing['preciocasa'])
    plt.show()

    # Representar las relaciones entre el precio de la casa y el resto de características (numéricas) para ver si hay alguna relación fuerte o no. Diagrama de dispersión. Ver la matriz de correlación.
    corr_matrix = housing.corr()
    median_house_value_correlations = corr_matrix['preciocasa']
    print(median_house_value_correlations)

    important_numerical_values = ['preciocasa', 'area', 'baños', 'pisos']

    scatter_matrix(housing[important_numerical_values], figsize=(20, 10))
    plt.show()

    # Representar diagramas de barras de las características de variables discretas y categóricas. Previamente habrá
    # que contar el número de registros por cada valor de la característica.

    parking_group = housing.groupby('parking').size()
    print(housing['parking'].unique())
    plt.bar(housing['parking'].unique(), parking_group)
    plt.xticks(housing['parking'].unique())
    plt.show()

    parking_group = housing.groupby('porche').size()
    print(housing['porche'].unique())
    plt.bar(housing['porche'].unique(), parking_group)
    plt.xticks(housing['porche'].unique())
    plt.show()

    # 3.	Preprocesamiento de los datos.
    print(housing[housing.duplicated()])
    print(housing.drop_duplicates().shape, housing.shape) # no hay duplicados para eliminar

    # Comprobar valores nulos y qué hacer con ellos.
    print(housing.info())  # vemos que no hay nulos

    # Convertir las categóricas a numéricas con one hot encoding
    one_encoder = ColumnTransformer([('one_hot', OneHotEncoder(), cat_columns)], sparse_threshold=0)
    categorical_values = one_encoder.fit_transform(housing[cat_columns])
    cat_column_transformer = list(map(lambda x: x.split('__')[1], one_encoder.get_feature_names_out()))
    categorical_values = pd.DataFrame(categorical_values, columns=cat_column_transformer)

    # Detectar outliers y borrarlos. Utilizar rango intercuartílico.
    q1 = np.quantile(housing.area, 0.25)
    q3 = np.quantile(housing.area, 0.75)

    k = 1.5

    RIQ = q3 - q1
    xL = q1 - k * RIQ
    xR = q3 + k * RIQ

    print('Rango Izquierda {}, Rango Derecho {}'.format(xL, xR))
    outliers = filter(lambda x: x[1] < xL or x[1] > xR, enumerate(housing.area))
    outliers = pd.DataFrame(list(map(lambda x: housing.iloc[x[0]], outliers)))
    print(outliers.index)
    housing.drop(index=outliers.index, inplace=True)

    # Estandarizar el dataset con la StandardScaler()
    standardScaler = StandardScaler()
    numerical_values = standardScaler.fit_transform(housing[num_columns])
    numerical_values = pd.DataFrame(numerical_values, columns=standardScaler.get_feature_names_out())

    # Crear el nuevo dataset ya preprocesado.
    housing_preprocessed = pd.concat([numerical_values, categorical_values], axis=1)
