import tensorflow as tf

if __name__ == '__main__':
    fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
    (x_train_full, y_train_full), (x_test, y_test) = fashion_mnist
    x_train, y_train = x_train_full[:-5000], y_train_full[:-5000]
    x_valid, y_valid = x_train_full[-5000:], y_train_full[-5000:]

    # now we scale the data to work better in neurons
    x_train, x_valid = x_train / 255, x_valid / 255

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'shirt', 'Sneaker', 'Bag',
                   'Ankle boot']

    tf.random.set_seed(42)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=x_train.shape[1:]))
    # we have to flat the data, because the input neuron of the network cant be a matrix
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(300, activation='relu'))  # hidden layer
    model.add(tf.keras.layers.Dense(100, activation='relu'))  # hidden layer
    model.add(tf.keras.layers.Dense(10, activation='softmax'))  # output layer

    model.summary()

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=30, validation_data=(x_valid, y_valid))

    print('model evaluation: ', model.evaluate(x_test, y_test))
