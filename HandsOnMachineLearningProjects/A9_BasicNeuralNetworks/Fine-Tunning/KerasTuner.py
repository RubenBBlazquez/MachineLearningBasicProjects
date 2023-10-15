import keras_tuner as kt
import tensorflow as tf
from keras_tuner import HyperModel
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


def build_model(hp: kt.HyperParameters):
    # if the parameter is already defined in the parameters it returns the number,
    # if not, it generates a new value between 0 and 8
    n_hidden = hp.Int('n_hidden', min_value=0, max_value=8, default=2)
    n_neurons = hp.Int('n_neurons', min_value=16, max_value=256)
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')

    optimizer = hp.Choice('optimizer', ['sgd', 'adam'])
    if optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())

    for _ in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_neurons, activation='relu'))

    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


# if you want to fine tune model.fit arguments such as batch_size
# , you could create a class that inherit the hyperModel from keras_tunner
class MyClassificationHyperModel(HyperModel):
    def build(self, hp):
        return build_model(hp)

    def fit(self, hp, model, *args, **kwargs):
        x, y = args

        if hp.Boolean("normalizer"):
            norm_layer = tf.keras.layers.Normalization()
            x = norm_layer(x)

        return model.fit(x, y, **kwargs)


# after we define the class, we can use it with a hyper_band_tunner
def get_random_hyper_band_tunner():
    return kt.Hyperband(
        MyClassificationHyperModel(), objective='val_accuracy', seed=42, max_epochs=10,
        factor=3, hyperband_iterations=2, overwrite=True, directory='kt_tunner_info', project_name='hyperband'
    )


def get_bayesian_hyper_band_tunner():
    return kt.BayesianOptimization(
        MyClassificationHyperModel(), objective='val_accuracy', seed=42, max_trials=10,
        alpha=1e-4, beta=2.6,
        overwrite=True, directory='kt_tunner_info', project_name='bayesian_tunner'
    )


if __name__ == '__main__':
    housing = fetch_california_housing()

    x_train, x_test, y_train, y_test = train_test_split(housing.data, housing.target, train_size=0.7)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, train_size=0.7)

    random_search_tuner = kt.RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=5,
        overwrite=True,
        directory='kt_tunner_info',
        project_name='my_rnd_search',
        seed=42
    )
    random_search_tuner.search(x_train, y_train, epochs=10, validation_data=(x_valid, y_valid))

    top3_models = random_search_tuner.get_best_models(num_models=3)
    best_model = top3_models[0]

    top3_params = random_search_tuner.get_best_hyperparameters(num_trials=3)
    best_params = top3_params[0].values

    # kt_tunner is guided by a so_called oracle, where it defines each trial and the next, to not repeat the same trial
    # you can access to that oracle from the directory that creates the method `kt_tunner_info`
    # or with...
    best_trial = random_search_tuner.oracle.get_best_trials(num_trials=1)[0]
    print('------------------------------------------------')
    print(best_trial.summary())
    print('-------------------------------------------------')

    # this method will train many models and selecting the best model
    my_tune_model = get_random_hyper_band_tunner()
    my_tune_model.search(x_train, y_train, epochs=10, validation_data=(x_valid, y_valid))
    best_random_tune_model = my_tune_model.get_best_models(1)

    # one thing to know if that HyperBand is smarter than pure random search in the way it allocates resources
    # Keras Tunner also includes a kt.BayesianOptimization tuner:
    # this algorithm gradually learns which region of the hyperparameters space are most promising
    # by fitting a probabilistic model called a Gaussian Process. This allows it to gradually zoom in on the best
    # hyperparameters.
    # The downside is that the algorithm has its own hyperparameters:
    # -> alpha, represents the level of noise you expect in the performance measures across trials (its default to 10-4)
    # -> beta, specifies how much you want the algorithm to explore, instead of simply exploiting the known good regions
    # of hyperparameters space (its default to 2.6)

    my_bayesian_tune_model = get_bayesian_hyper_band_tunner()
    my_bayesian_tune_model.search(x_train, y_train, epochs=10, validation_data=(x_valid, y_valid))
    best_bayesian_model = my_tune_model.get_best_models(1)

