import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

if __name__ == '__main__':
    housing = pd.read_csv('../files/housing/housing.csv')
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
    # plt.show()

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
