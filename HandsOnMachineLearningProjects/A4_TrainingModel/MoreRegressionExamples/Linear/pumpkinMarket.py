from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

if __name__ == '__main__':
    pumpkinInformation = pd.read_csv('../datasets/pumpkinMarketUS.csv', delimiter=',')
    print(pumpkinInformation.head())
    print()
    print(pumpkinInformation.isnull().sum())  # we check how many null values we have

    # so, we will get only columns that not have null values to do dataframe easier
    columns_without_nulls = ['Package', 'Month', 'Low Price', 'High Price', 'Date', 'City Name', 'Variety']
    pumpkins = pumpkinInformation.drop(
        [column for column in pumpkinInformation.columns if column not in columns_without_nulls],
        axis=1)
    pumpkins = pumpkins[pumpkins['Package'].str.contains('bushel', case=True, regex=True)]

    average_pumpkin_prices = (pumpkins['Low Price'] + pumpkins['High Price']) / 2
    month = pd.DatetimeIndex(pumpkins['Date']).month
    day_of_the_year = pd.to_datetime(pumpkins['Date']).apply(lambda x: (x - datetime(x.year, 1, 1)).days)

    new_pumpkins = pd.DataFrame({'Day Of The Year': day_of_the_year, 'Month': month, 'Package': pumpkins['Package'],
                                 'Low Price': pumpkins['Low Price'],
                                 'High Price': pumpkins['High Price'], 'Variety': pumpkins['Variety'],
                                 'Price': average_pumpkin_prices, 'City Name': pumpkins['City Name']})

    # in my opinion must be a multiplication because we have 1 1/9 bushel and 1/2 bushel,
    # so, the price must be in first case the sum of 1 bushel and 1/9 and second the half of it
    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1 1/9'), 'Price'] = average_pumpkin_prices / (1 + 1 / 9)
    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1/2'), 'Price'] = average_pumpkin_prices / (1 / 2)

    price = new_pumpkins.Price
    month = new_pumpkins.Month
    plt.scatter(price, month)
    plt.show()

    new_pumpkins.groupby(['Month'])['Price'].mean().plot(kind='bar')
    plt.ylabel("Pumpkin Price")
    plt.xticks(rotation=360)
    plt.show()

    new_pumpkins.dropna(inplace=True)

    new_pumpkin_dataframe = new_pumpkins['Month']
    new_pumpkin_dataframe = pd.concat([new_pumpkin_dataframe, pd.get_dummies(new_pumpkins['Variety'])], axis=1)

    scaler = StandardScaler()
    scaler.fit(new_pumpkin_dataframe)
    scaled_data = scaler.transform(new_pumpkin_dataframe)

    X_train, X_test, y_train, y_test = train_test_split(scaled_data, new_pumpkins['Price'], test_size=0.2,
                                                        random_state=0)

    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    summary_coefficients = pd.DataFrame({'Columns': new_pumpkin_dataframe.columns.values,
                                         'Coefficients': lin_reg.coef_})

    print(summary_coefficients)

    pred = lin_reg.predict(X_test)

    mse = np.sqrt(mean_squared_error(y_test, pred))
    print(f'Mean error: {mse:3.3} ({mse / np.mean(pred) * 100:3.3}%)')

    score = lin_reg.score(X_train, y_train)
    # you can use score function or
    # r_squared = 1 - (np.sum((y_train - pred) ** 2) / np.sum((y_train - np.mean(y_train)) ** 2))
    print('Model determination R2: ', score)

    prediction_dataframe = pd.DataFrame(X_test,
                                        columns=['Month', 'Fairy Tale', 'Miniature', 'Mixed Heirloom', 'Pie Type'])
    # prediction_dataframe['F:Month (y = b+mx)'] = lin_reg.intercept_ + (lin_reg.coef_[0] * prediction_dataframe['Month'])
    # prediction_dataframe['F:Day (y = b+mx)'] = lin_reg.intercept_ + (
    #        lin_reg.coef_[1] * prediction_dataframe['Day Of The Year'])
    prediction_dataframe['Predicted Price'] = pred
    prediction_dataframe['Test Price'] = y_test.reset_index().drop('index', axis=1)

    variety_labels = pd.DataFrame(prediction_dataframe[['Fairy Tale', 'Miniature', 'Mixed Heirloom', 'Pie Type']])
    variety_labels = variety_labels.idxmax(axis=1)
    label_mapping = {'Fairy Tale': 0, 'Miniature': 1, 'Mixed Heirloom': 2, 'Pie Type': 3}
    colors = ['red', 'green', 'blue', 'orange']
    variety_colors = [colors[label_mapping[label]] for label in variety_labels]

    plt.scatter(prediction_dataframe['Month'], y_test, c=variety_colors)

    # Add the predictions as a line plot
    order = np.lexsort([pred, prediction_dataframe['Month']])
    plt.plot(prediction_dataframe['Month'][order], pred[order], c='red')

    # Add labels and title
    plt.xlabel('Month')
    plt.ylabel('Price')
    plt.title('Price of Pumpkins by Variety and Month (Test Set)')

    # Show the plot
    plt.show()
