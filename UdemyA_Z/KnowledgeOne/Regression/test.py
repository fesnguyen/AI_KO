from cr_regression import *

df = pd.read_csv('../Data/house_price_train.csv')

line_regression(df, 'MedHouseVal')
