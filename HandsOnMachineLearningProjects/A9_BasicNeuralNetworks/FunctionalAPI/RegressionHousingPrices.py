import numpy as np

import pandas as pd
import tensorflow as tf
from keras import Input
from keras.layers import Normalization, Dense, Concatenate
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

if __name__ == '__main__':
    housing = pd.read_csv('../../files/housing/housing.csv')

    ord_encoder = OrdinalEncoder()
    housing['ocean_proximity'] = ord_encoder.fit_transform(housing['ocean_proximity'].to_numpy().reshape(-1, 1))

    inputer = SimpleImputer(strategy='mean')
    housing_inputer = inputer.fit_transform(housing)
    housing.loc[:] = housing_inputer

    target = housing['median_house_value']
    x = housing.drop('median_house_value', axis=1).astype(np.float32)

    x_train, x_test, y_train, y_test = train_test_split(x, target, train_size=0.7)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, train_size=0.7)

    normalization_layer = Normalization()
    hidden_layer = Dense(30, activation='relu')
    hidden_layer_2 = Dense(30, activation='relu')
    concat_layer = Concatenate()
    output_layer = Dense(1)

    # now we use the functional API, passing the previous layer like a function
    input_layer = Input(shape=x_train.shape[1:])
    normalized = normalization_layer(input_layer)
    hidden1 = hidden_layer(normalized)
    hidden2 = hidden_layer_2(hidden1)
    concat = concat_layer([normalized, hidden2])
    output = output_layer(concat)

    model = tf.keras.Model(inputs=[input_layer], outputs=[output])

    model.compile(loss='mse', optimizer='adam', metrics=['RootMeanSquaredError'])

    model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=200)

    prediction = model.predict(x_test.iloc[0].to_numpy().reshape(-1))
    print(f'test prediction,  real_value: {y_test.iloc[0]}, prediction: {prediction}')
