import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
# alternative method: from sklearn.tree import DecisionTreeRegressor

file_path = '../train.csv'
home_data = pd.read_csv(file_path)

y = home_data.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

# Split the data into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)

randomforest_model = RandomForestRegressor(random_state = 1)
randomforest_model.fit(train_X, train_y)
rf_val_mae = mean_absolute_error(randomforest_model.predict(val_X), val_y)
print(f"Validation MAE for Random Forest Model is: {rf_val_mae}")

# Now train the model on all training data
whole_randomforest_model = RandomForestRegressor(random_state = 1)
whole_randomforest_model.fit(X, y)

test_data_path = '../test.csv'
test_data = pd.read_csv(test_data_path)
test_X = test_data[features]
test_predictions = whole_randomforest_model.predict(test_X)

output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': test_predictions})
print(output)
