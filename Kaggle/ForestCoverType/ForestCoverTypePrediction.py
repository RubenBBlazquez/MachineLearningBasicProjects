import pandas as pd
import numpy as np
from keras.layers import Dense, Dropout
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.svm import SVC
from keras import Sequential


def preprocess_data(dataframe):
    scaler = StandardScaler()

    Ids = dataframe['Id']
    target = None
    columns_to_drop = []

    if 'Cover_Type' in dataframe.columns:
        target = dataframe['Cover_Type']
        columns_to_drop.append('Cover_Type')

    dataframe = dataframe.copy().drop([*columns_to_drop, 'Id'], axis=1)

    one_hot_columns = []
    non_one_hot_columns = []

    for index, column_data in dataframe.items():
        if set(sorted(column_data.unique())) == {0, 1} or set(sorted(column_data.unique())) == {0} or set(
                sorted(column_data.unique())) == {1}:
            one_hot_columns.append(column_data.name)
        else:
            non_one_hot_columns.append(column_data.name)

    df_with_one_hot_columns = dataframe[one_hot_columns]
    df_with_non_one_hot_columns = dataframe[non_one_hot_columns]
    df_with_non_one_hot_columns = pd.DataFrame(scaler.fit_transform(df_with_non_one_hot_columns),
                                               columns=non_one_hot_columns)

    final_dataframe = pd.concat([df_with_non_one_hot_columns, df_with_one_hot_columns], axis=1)

    if target is not None:
        pass
        # selector = SelectKBest(chi2, k=5)  # Select top 5 features
        # data = selector.fit_transform(final_dataframe, target)
        # print(final_dataframe.columns[selector.get_support(indices=True)])
        # negative_correlations_with_target = pd.concat([final_dataframe, target], axis=1).corr()['Cover_Type'] < 0
        # final_dataframe = dataframe.loc[:, negative_correlations_with_target.values[:-1]]

    return final_dataframe, target, Ids


if __name__ == '__main__':
    train_data = pd.read_csv('files/train.csv', engine='pyarrow')
    test_data = pd.read_csv('files/test.csv', engine='pyarrow')

    cover_types_dict = {1: 'Spruce/Fir', 2: 'Lodgepole Pine', 3: 'Ponderosa Pine', 4: 'Cottonwood/Willow', 5: 'Aspen',
                        6: 'Douglas-fir', 7: 'Krummholdz'}

    train_dataframe, target, ids = preprocess_data(train_data)

    train_x, test_x, train_y, test_y = train_test_split(train_dataframe, target, random_state=42, train_size=0.8)

    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(train_x, train_y)
    print('Logistic Regression With best Correlation Attrs: ', log_reg.score(test_x, test_y))

    rdn_forest = RandomForestClassifier(random_state=42)
    rdn_forest.fit(train_x, train_y)

    print('Random Forests With best Correlation Attrs: ', rdn_forest.score(test_x, test_y))

    svc = SVC(random_state=42, coef0=1, C=5)
    svc.fit(train_x, train_y)
    print('SVC With best Correlation Attrs: ', svc.score(test_x, test_y))

    sgd = SGDClassifier(random_state=42)
    sgd.fit(train_x, train_y)
    print('SGDClassifier With best Correlation Attrs: ', sgd.score(test_x, test_y))

    one_hot = OneHotEncoder(sparse_output=False)

    test_y = one_hot.fit_transform(test_y.to_numpy().reshape(-1, 1))
    train_y = one_hot.fit_transform(train_y.to_numpy().reshape(-1, 1))

    neural_network = Sequential()
    neural_network.add(Dense(27, activation='relu'))
    neural_network.add(Dense(50, activation='relu'))
    neural_network.add(Dense(len(cover_types_dict.keys()), activation='softmax'))

    neural_network.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['Accuracy'])
    neural_network.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=100)

    test_data, target, ids = preprocess_data(test_data)
    predictions = rdn_forest.predict(test_data)
    predictions = list(map(lambda x: {'Id': x[0], 'Cover_Type': x[1].argmax() +1}, zip(ids, predictions)))
    predictions = pd.DataFrame(predictions)

    predictions.to_csv('files/submission.csv', index=False)
