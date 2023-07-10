from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    housing = fetch_california_housing()
    x_train_full, x_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train_full, y_train_full)

    mlp_reg = MLPRegressor(hidden_layer_sizes=[50, 50, 50], random_state=42)
    pipeline = make_pipeline(StandardScaler(), mlp_reg)
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_valid)

    # if your data have a lot of outliers you could use the mean absolute error for a better performance
    # if not, RMSE, is a good option
    rmse = mean_squared_error(y_valid, y_pred, squared=False)
    mae = mean_absolute_error(y_valid, y_pred)
    print('rmse', rmse)
    print('mae', mae)
