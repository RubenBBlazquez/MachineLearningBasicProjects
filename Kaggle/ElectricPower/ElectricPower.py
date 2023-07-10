import pandas as pd

if __name__ == '__main__':
    main_df = pd.read_csv('files/train_df.csv')

    dependent_variables = main_df.drop('POWER', axis=1)
    target = main_df['POWER']

    correlations = main_df.corr()
