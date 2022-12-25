import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.compose import make_column_selector, make_column_transformer, ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from HandsOnMachineLearningProjects.A2_CustomTransformers.ClusterSimilarity import ClusterSimilarity
import time


def column_ratio(x):
    return x[:, [0]] / x[:, [1]]


def ratio_name(function_transformer, feature_names_in):
    return ['ratio']


def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy='median'),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler()
    )


if __name__ == '__main__':
    original_housing = pd.read_csv('../files/housing/housing.csv')
    housing = original_housing.copy().drop('median_house_value', axis=1)
    housing_labeled = original_housing['median_house_value']
    print(housing.describe())

    housing.hist(grid=True, bins=50)
    # plt.show()

    # we use SimpleImputer to fill null values with median, and after that we scale values
    num_pipeline = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler()
    )

    # if we have to encoder categorical values we will fill nun values with the most frequent
    # and after that we use oneHotEncoder to transform in dummies
    categorical_pipeline = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder()
    )

    preprocessing = make_column_transformer(
        (num_pipeline, make_column_selector(dtype_include=np.number)),
        (categorical_pipeline, make_column_selector(dtype_include='object'))
    )

    housing_with_make_column_array = preprocessing.fit_transform(original_housing)
    name_columns = list(map(lambda x: x.split('__')[1], preprocessing.get_feature_names_out()))
    housing_with_make_column = pd.DataFrame(housing_with_make_column_array, columns=name_columns)

    log_pipeline = make_pipeline(
        SimpleImputer(strategy='median'),
        FunctionTransformer(np.log, feature_names_out='one-to-one'),
        StandardScaler()
    )

    cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1, random_state=42)

    default_num_pipeline = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())

    preprocessing = ColumnTransformer([
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline(), ["population", "households"]),
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                               "households", "median_income"]),
        ("geo", cluster_simil, ["latitude", "longitude"]),
        ("cat", categorical_pipeline, make_column_selector(dtype_include=object)),
    ], remainder=default_num_pipeline)

    housing_preprocessed = pd.DataFrame(preprocessing.fit_transform(original_housing),
                                        columns=preprocessing.get_feature_names_out())

    full_pipeline = Pipeline([
        ('preprocessing', preprocessing),
        ('random_forest', RandomForestRegressor(random_state=42))
    ])

    param_grid = [
        {'preprocessing__geo__n_clusters': [5, 8, 10],
         'random_forest__max_features': [4, 6, 8]},
        {'preprocessing__geo__n_clusters': [10, 15],
         'random_forest__max_features': [6, 8, 10]},
    ]

    execution_before = time.time()

    # we use n_jobs=-1 to use all available cores
    grid_search = GridSearchCV(full_pipeline, param_grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=5)
    grid_search.fit(housing, housing_labeled)
    execution_after = time.time()

    print(grid_search.best_params_, 'time: ', (execution_after - execution_before))

    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results.sort_values(by='mean_test_score', ascending=False, inplace=True)
    print(cv_results.head())
