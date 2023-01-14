import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.feature_selection import mutual_info_classif, SelectKBest, chi2

if __name__ == '__main__':
    train_dataframe = pd.read_csv('datasets/train.csv')
    print(train_dataframe.head())
    print(train_dataframe.info())

    print(train_dataframe['Destination'].unique())
    print(train_dataframe['HomePlanet'].unique())

    train_dataframe_2 = train_dataframe.copy()
    var_to_predict = train_dataframe['Transported'].map({True: 1, False: 0})

    train_dataframe_2.drop(['Transported'], axis=1, inplace=True)

    # now we drop non dependent variables
    train_dataframe_2.drop(['PassengerId', 'Name', 'Cabin'], axis=1, inplace=True)

    num_pipeline = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler()
    )

    cat_ohe_pipeline = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder()
    )

    cat_oe_pipeline = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OrdinalEncoder()
    )

    transformer = ColumnTransformer([
        ('num_pipeline', num_pipeline, make_column_selector(dtype_include=np.number)),
        ('cat_oe_pipeline', cat_oe_pipeline, ['HomePlanet', 'Destination']),
        ('cat_ohe_pipeline', cat_ohe_pipeline, ['CryoSleep', 'VIP', ]),
    ], sparse_threshold=0)

    transformer2 = ColumnTransformer([
        ('num_pipeline', num_pipeline, make_column_selector(dtype_include=np.number)),
        ('cat_pipeline', cat_ohe_pipeline, make_column_selector(dtype_include=object)),
    ], sparse_threshold=0)

    scaled_info = transformer2.fit_transform(train_dataframe_2)
    columns_scaled_info = list(map(lambda x: x.split('__')[1], transformer2.get_feature_names_out()))
    train_preprocessed = pd.DataFrame(scaled_info, columns=columns_scaled_info)

    train_x, train_test_x, train_y, train_test_y = train_test_split(train_preprocessed, var_to_predict,
                                                                    random_state=42, train_size=0.8)

    log_reg = LogisticRegression(random_state=42)
    random_forest = RandomForestClassifier(random_state=42)
    sgd_clf = SGDClassifier(random_state=42)
    k_clf = KNeighborsClassifier(n_neighbors=12)

    print('logistic regression: ',
          cross_val_score(log_reg, train_test_x, train_test_y, scoring='accuracy', cv=10, n_jobs=-1))
    print('Random forest: ', cross_val_score(random_forest, train_test_x, train_test_y, cv=10, n_jobs=-1))
    print('SGD Classifier', cross_val_score(sgd_clf, train_test_x, train_test_y, cv=10, n_jobs=-1))
    print('K Classifier', cross_val_score(k_clf, train_test_x, train_test_y, cv=10, n_jobs=-1))

    kf = KFold(n_splits=10)

    # Initialize lists to store the results
    train_scores = []
    test_scores = []
    train_data = []
    test_data = []

    # Iterate over the splits
    for train_index, test_index in kf.split(train_preprocessed):
        X_train, X_test = train_preprocessed.iloc[train_index], train_preprocessed.iloc[test_index]
        y_train, y_test = var_to_predict[train_index], var_to_predict[test_index]
        train_data.append((X_train, y_train))
        test_data.append((X_test, y_test))

        random_forest.fit(X_train, y_train)
        train_scores.append(random_forest.score(X_train, y_train))
        test_scores.append(random_forest.score(X_test, y_test))

    index_test = np.array(test_scores).argmax()
    print(index_test)
    print(len(train_data[index_test][0]))
    print(len(test_data[index_test][0]))
    random_forest2 = RandomForestClassifier(random_state=42, n_jobs=-1)
    random_forest2.fit(train_data[index_test][0], train_data[index_test][1])
    print(random_forest2.score(test_data[index_test][0], test_data[index_test][1]))

    predictions = random_forest2.predict(test_data[index_test][0])

    test_data = pd.read_csv('datasets/test.csv')
    test_data_copy = test_data.drop(['PassengerId', 'Name', 'Cabin'], axis=1)

    scaled_info_test = transformer2.fit_transform(test_data_copy)
    columns_scaled_info = list(map(lambda x: x.split('__')[1], transformer2.get_feature_names_out()))
    test_preprocessed = pd.DataFrame(scaled_info_test, columns=columns_scaled_info)

    predictions_test = random_forest2.predict(scaled_info_test)

    print(len(predictions_test), len(test_data['PassengerId']))
    pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Transported': predictions_test == 1 }).to_csv(
        'datasets/submission.csv', index=False)


