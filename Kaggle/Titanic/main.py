import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.compose import make_column_transformer, make_column_selector, ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, MinMaxScaler


class PreProcessTitanicDataframe:
    def __init__(self, dataframe):
        self.titanic_dataframe = dataframe
        self.preprocessing = None

        self.fill_columns_nan_values()
        print(self.titanic_dataframe.info())
        self.remove_not_independent_variables()

        self.titanic_dataframe_scaled = self.normalize_independent_variables()

    def fill_columns_nan_values(self):
        ages = self.titanic_dataframe['Age'][:, np.newaxis]
        embarks = self.titanic_dataframe.loc[:, 'Embarked']
        median_transformer = SimpleImputer(strategy='mean')
        most_frequent_transformer = SimpleImputer(strategy='most_frequent')
        self.titanic_dataframe['Age'] = pd.Series(median_transformer.fit_transform(ages).reshape(1, -1)[0])
        self.titanic_dataframe['isChild'] = (self.titanic_dataframe['Age'] <= 8).astype(np.dtype(int))
        self.titanic_dataframe['isRich'] = (self.titanic_dataframe['Fare'] > 50).astype(np.dtype(int))
        self.titanic_dataframe['Cabin'][self.titanic_dataframe['Cabin'].isna()] = 'non_cabin'
        self.titanic_dataframe['Embarked'] = \
            pd.Series(most_frequent_transformer.fit_transform(embarks[:, np.newaxis]).reshape(1, -1)[0])

    def remove_not_independent_variables(self):
        self.titanic_dataframe.drop(['PassengerId', 'Name', 'Ticket'],
                                    inplace=True, axis=1)

    def normalize_independent_variables(self):
        num_pipeline = make_pipeline(
            OrdinalEncoder(),
            StandardScaler()
        )

        self.preprocessing = ColumnTransformer([
            ('embarked', OneHotEncoder(), ["Embarked"]),
            ('sex', OneHotEncoder(), ["Sex"]),
            ('pclass', OneHotEncoder(), ["Pclass"]),
        ])

        scaled = self.preprocessing.fit_transform(self.titanic_dataframe)
        names = list(map(lambda x: x.split('__')[1], self.preprocessing.get_feature_names_out()))

        scaled = pd.DataFrame(scaled, columns=names)
        scaled['isChild'] = self.titanic_dataframe['isChild']
        scaled['isRich'] = self.titanic_dataframe['isRich']
        scaled['SibSp'] = self.titanic_dataframe['SibSp']
        scaled['Parch'] = self.titanic_dataframe['Parch']

        return scaled


if __name__ == '__main__':
    titanic_dataframe_train = pd.read_csv('datasets/train.csv')
    titanic_dataframe_test = pd.read_csv('datasets/test.csv')
    submission_example = pd.read_csv('datasets/gender_submission.csv')

    values_to_predict = titanic_dataframe_train['Survived']
    titanic_dataframe_train.drop(['Survived'], axis=1, inplace=True)

    x_train, x_text, y_train, y_test = train_test_split(titanic_dataframe_train, values_to_predict, random_state=42,
                                                        train_size=0.80)

    x_train.reset_index(inplace=True)
    x_train.drop('index', axis=1, inplace=True)
    preprocessing = PreProcessTitanicDataframe(titanic_dataframe_train)
    scaled_inputs = preprocessing.titanic_dataframe_scaled

    x_text.reset_index(inplace=True)
    x_text.drop('index', axis=1, inplace=True)
    titanic_dataframe_test_preprocessed = titanic_dataframe_test.copy()
    preprocessing = PreProcessTitanicDataframe(titanic_dataframe_test_preprocessed)
    scaled_inputs_tests = preprocessing.titanic_dataframe_scaled

    # preprocessing
    dummy_classifier = DummyClassifier(random_state=42)

    sgd_classifier = SGDClassifier(random_state=42)

    forest_classifier = RandomForestClassifier()

    log_reg = LogisticRegression(random_state=42)

    forest_classifier.fit(scaled_inputs, values_to_predict)
    predictions = forest_classifier.predict(scaled_inputs_tests)

    print(cross_val_score(sgd_classifier, scaled_inputs, values_to_predict, cv=10, n_jobs=-1).mean())
    print(cross_val_score(forest_classifier, scaled_inputs, values_to_predict, cv=10, n_jobs=-1).mean())
    print(cross_val_score(log_reg, scaled_inputs, values_to_predict, cv=10, n_jobs=-1,scoring='accuracy').mean())


    submission = pd.DataFrame({'PassengerId': titanic_dataframe_test['PassengerId'], 'Survived': predictions})
    submission.to_csv('datasets/submission.csv', index=False)
