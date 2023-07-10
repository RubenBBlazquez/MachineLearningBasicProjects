import numpy as np
import pandas as pd
from keras import Input, Model
from keras.layers import Dense, concatenate, Normalization
from sklearn.datasets import fetch_california_housing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

# Use Cases in which you may want to have multiple outputs:
#
# --- The Task may demand it. For Instance, you may want to classify the main object in a picture.
# This is both regression task and a classification task
#
# --- Similarly, you may have multiple independent tasks based on the same data. Sure, you could train one
# neural network per task, but in many cases you will get better results on all tasks by training one. This is
# because the neural network can learn features in the data that are useful across tasks. For example, you could
# perform multitask classification on picture of faces, using one output to classify the persons facial expressions
# (smiling, surprising, etc..), and another output to identify whether they are wearing glasses or not

if __name__ == '__main__':
    housing = fetch_california_housing()

    x_train, x_test, y_train, y_test = train_test_split(housing.data, housing.target, train_size=0.7)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, train_size=0.7)

    input_wide = Input(shape=[5])  # features 0 to 4
    input_deep = Input(shape=[6])  # features 2 to 7

    norm_layer_wide = Normalization()
    norm_layer_deep = Normalization()

    norm_wide = norm_layer_wide(input_wide)
    norm_deep = norm_layer_wide(input_wide)

    hidden1 = Dense(30, activation='relu')(norm_deep)
    hidden2 = Dense(30, activation='relu')(hidden1)

    concat = concatenate([norm_wide, hidden2])

    output = Dense(1)(concat)

    model = Model(inputs=[input_wide, input_deep], outputs=[output])
    model.compile(loss='mse', optimizer='adam', metrics=['RootMeanSquaredError'])

    x_train_wide, x_train_deep = x_train[:, :5], x_train[:, 2:]
    x_valid_wide, x_valid_deep = x_valid[:, :5], x_valid[:, 2:]
    x_test_wide, x_test_deep = x_test[:, :5], x_test[:, 2:]
    x_new_wide, x_new_deep = x_test_wide[:3], x_test_deep[:3]

    norm_layer_deep.adapt(x_train_deep)
    norm_layer_wide.adapt(x_train_wide)

    history = model.fit((x_train_wide, x_train_deep), y_train, epochs=20,
                        validation_data=((x_valid_wide, x_valid_deep), x_valid))
    mse_test = model.evaluate((x_test_wide, x_test_deep), y_test)

    print('loss - rmse', mse_test)
    print(model.predict((x_new_wide, x_new_deep)))
