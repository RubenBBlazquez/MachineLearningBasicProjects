import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score, GridSearchCV
from pandas.plotting import scatter_matrix
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.tree import DecisionTreeRegressor

from HandsOnMachineLearningProjects.A2_CustomTransformers.ClusterSimilarity import ClusterSimilarity


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
    housing = original_housing.copy()
    print(housing.describe())

    housing.hist(bins=50, figsize=(12, 8))
    plt.show()

    housing['income_cat'] = pd.cut(housing['median_income'], bins=[0, 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=[1, 2, 3, 4, 5])
    housing['income_cat'].value_counts().sort_index().plot.bar(rot=0, grid=True)
    plt.xlabel('Income Category')
    plt.ylabel('Number Of Districts')
    # plt.show()

    splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    strata_splits = []

    for train_index, test_index in splitter.split(housing, housing['income_cat']):
        strata_train_set_n = housing.iloc[train_index]
        strata_test_set_n = housing.iloc[test_index]
        strata_splits.append([strata_train_set_n, strata_test_set_n])

    strata_train_set, strata_test_set = train_test_split(housing, test_size=0.2, stratify=housing['income_cat'],
                                                         random_state=42)

    print(strata_test_set['income_cat'].value_counts() / len(strata_test_set))

    for set_ in (strata_train_set, strata_test_set):
        set_.drop('income_cat', axis=1, inplace=True)

    housing = pd.DataFrame(strata_train_set.copy())

    housing.plot(kind='scatter', x='longitude', y='latitude', s=housing['population'] / 100,
                 cmap='jet', label='population', colorbar=True, c='median_house_value', sharex=False
                 , figsize=(10, 7))
    plt.show()

    # we get correlation matrix to know how attributes correlates with median_housing_value
    # (we correlate with median_house_value because we want to predict if sth is a good investment
    # so, the more valuable attribute is median_house_value)#
    corr_matrix = housing.corr()
    median_house_value_correlations = corr_matrix['median_house_value']
    print(median_house_value_correlations)

    best_correlation_attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
    scatter_matrix(housing[best_correlation_attributes], figsize=(20, 10), grid=True)
    plt.show()

    housing.plot(kind='scatter', x='median_income', y='median_house_value')
    # plt.show()

    # now we are going to clean data, so, first we can want to know how many rooms per house are
    housing['rooms_per_house'] = housing['total_rooms'] / housing['households']

    # or know the number of bedrooms per room
    housing['bedroom_per_room'] = housing['total_bedrooms'] / housing['total_rooms']

    # or know how many people live in each house
    housing['people_per_household'] = housing['population'] / housing['households']

    # we get the correlations again to know if attributes calculated recently means something
    corr_matrix = housing.corr()
    median_house_value_correlations = corr_matrix['median_house_value']
    print(median_house_value_correlations.sort_values(ascending=False))

    best_correlation_attributes = ['median_house_value', 'median_income', 'housing_median_age',
                                   'rooms_per_house', 'bedroom_per_room']
    scatter_matrix(housing[best_correlation_attributes], figsize=(20, 10), grid=True)
    # plt.show()

    # start cleaning, first we are going to separate train data and predict data(in our case median_house_value) so...
    housing = pd.DataFrame(strata_train_set.drop(['median_house_value'], axis=1))
    housing_labeled = strata_train_set['median_house_value'].copy()

    # we fill nan values with median
    simple_inputer = SimpleImputer(strategy='median')
    housing_num = housing.select_dtypes(include=[np.number])
    simple_inputer.fit(housing_num)

    print('median_values = ', simple_inputer.statistics_, housing_num.index)

    new_housing = pd.DataFrame(simple_inputer.transform(housing_num), columns=housing_num.columns.values)
    print(new_housing.info())

    # clean categorical attributes(we use ordinal encoder to set one number such as 1,2,3... to category types)
    housing_cat = housing[['ocean_proximity']]
    # ordinal_encoder = OrdinalEncoder()
    # housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
    # to use ordinal Encoder Variables must have a correlation between them, so ,
    # we will use oneHotEncoder(to encoder variables in dummies)

    one_hot_encoder = OneHotEncoder()
    housing_cat_1hot = one_hot_encoder.fit_transform(housing_cat)
    print(housing_cat_1hot.toarray()[:8])
    print('We Can Get Categories Encoded From OneHotEncoder -> ', one_hot_encoder.categories_[0])
    print(len(new_housing), len(housing_cat_1hot.toarray()))
    ocean_proximity_df = pd.DataFrame(housing_cat_1hot.toarray(), columns=one_hot_encoder.get_feature_names_out())
    print(ocean_proximity_df.info())
    new_housing_with_dummies = pd.concat([new_housing, ocean_proximity_df], axis=1)

    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    housing_with_min_max_scaler = min_max_scaler.fit_transform(new_housing_with_dummies)
    print(housing_with_min_max_scaler)
    print()
    standard_scaler = StandardScaler()
    housing_with_standard_scaler = standard_scaler.fit_transform(new_housing_with_dummies)
    print(housing_with_standard_scaler)

    ages = np.linspace(new_housing_with_dummies["housing_median_age"].min(),
                       new_housing_with_dummies["housing_median_age"].max(),
                       500).reshape(-1, 1)

    # now we use rbf_kernel to use radial_basis_function to cut ages in a group in a range of age 35
    age_simil_35 = rbf_kernel(ages, [[35]], gamma=0.1)

    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Housing median age")
    ax1.set_ylabel("Number of districts")
    ax1.hist(housing["housing_median_age"], bins=50)
    ax2 = ax1.twinx()  # create a twin axis that shares the same x-axis
    color = "blue"
    ax2.plot(ages, age_simil_35, color=color, label="gamma = 0.10")
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylabel("Age similarity", color=color)

    plt.legend(loc="upper left")
    plt.show()

    df_with_ages_between_30_and_40 = pd.DataFrame(strata_train_set.loc[
                                                      (strata_train_set['housing_median_age'] >= 30) & (
                                                              strata_train_set['housing_median_age'] <= 40)])

    df_with_ages_between_30_and_40 = df_with_ages_between_30_and_40.reset_index()
    df_with_ages_between_30_and_40.drop('index', axis=1, inplace=True)
    housing_median_age = df_with_ages_between_30_and_40['housing_median_age']
    prices = df_with_ages_between_30_and_40['median_house_value']

    order = np.lexsort([prices, housing_median_age])
    plt.scatter(housing_median_age[order], prices[order])
    # plt.show()

    # we can use TransformedTargetRegressor to use a linear regression and scaler in the same fit
    some_new_data = housing[['median_income']].iloc[:5]
    model = TransformedTargetRegressor(LinearRegression(), transformer=StandardScaler())
    model.fit(housing[['median_income']], housing_labeled)
    predictions = model.predict(some_new_data)

    print(predictions)

    # now we use a transformer to transform data into logarithms,
    # and if you use inverse_transform you get the exponential
    log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
    log_pop = log_transformer.transform(housing[['population']])

    # also we can use Function Transformer to use the radial basis function mentioned before
    rbf_transformer = FunctionTransformer(rbf_kernel, kw_args=dict(Y=[[35]], gamma=0.1))
    age_simil_35_transformer = rbf_transformer.transform(ages)
    # print(age_simil_35[:10], age_simil_35_transformer[0:10])

    # use radial basis function with 2D array,
    # in this case this method will do the euclidian measure between the two variable
    sf_coords = 37.7749, -122.41
    rbf_transformer = FunctionTransformer(rbf_kernel, kw_args=dict(Y=[sf_coords], gamma=0.1))
    sf_simil = rbf_transformer.transform(housing[['latitude', 'longitude']])

    # use multiple functions
    array = np.array([[1, 2], [3, 4]])
    rbf_transformer = FunctionTransformer(lambda x: x[:, [0]] / x[:, [1]])

    print(rbf_transformer.transform(array))

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

    housing_preprocessed = pd.DataFrame(preprocessing.fit_transform(housing),
                                        columns=preprocessing.get_feature_names_out())

    lin_reg = make_pipeline(preprocessing, LinearRegression())
    lin_reg.fit(housing, housing_labeled)
    predictions = lin_reg.predict(housing)

    print(predictions[:5].round(2), '\n', housing_labeled[:5].values.round(2))

    # model determination
    lin_rmse = mean_squared_error(housing_labeled, predictions, squared=False)
    print(lin_rmse)

    tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor())
    tree_reg.fit(housing, housing_labeled)
    predictions = tree_reg.predict(housing)

    # model determination
    tree_rmse = mean_squared_error(housing_labeled, predictions, squared=False)
    print(11, tree_rmse)  # this produces an overfitting (its remember data used when we trained the model )

    # we use this method to create 10 folds to validate the model and compare 9 to 1 with each fold
    tree_rmses = -cross_val_score(tree_reg, housing, housing_labeled, scoring='neg_root_mean_squared_error', cv=10)

    print('model determination Decission tree regressor', tree_rmses)

    forest_reg = make_pipeline(preprocessing, RandomForestRegressor(random_state=42))

    forest_rmses = -cross_val_score(forest_reg, housing, housing_labeled, scoring='neg_root_mean_squared_error', cv=10)
    print('model determination Forest regressor', pd.Series(forest_rmses).describe())

    forest_reg.fit(housing, housing_labeled)
    housing_predictions = forest_reg.predict(housing)
    forest_rmse = mean_squared_error(housing_labeled, housing_predictions,
                                     squared=False)

    print('Model error ', forest_rmse)

    full_pipeline = Pipeline([
        ('preprocessing', preprocessing),
        ('random_forest', RandomForestRegressor)
    ])

    param_grid = [
        {'preprocessing__geo__n_clusters': [5, 8, 10],
         'random_forest__max_features': [4, 6, 8]},
        {'preprocessing__geo__n_clusters': [10, 15],
         'random_forest__max_features': [6, 8, 10]},
    ]

    grid_search = GridSearchCV(full_pipeline, param_grid, cv=3, scoring='root_mean_squared_error')
    grid_search.fit(housing, housing_labeled)

    print(grid_search.best_params_)


