import pandas as pd
import datetime
import numpy as np
from pyomo.environ import *
from pyomo.environ import value
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

amount_of_days = 200
df_production = pd.read_csv("C:\\Users\\thibo\\OneDrive - Hanzehogeschool Groningen\\Documenten\\Github\\UNAL_thesis\\xm_API\\production\\production_only_totals.csv")
df_consumption = pd.read_csv("C:\\Users\\thibo\\OneDrive - Hanzehogeschool Groningen\\Documenten\\Github\\UNAL_thesis\\xm_API\\consumption\\consumption_varia.csv")
df_consumption = df_consumption.drop(columns=['Unnamed: 0','DemaCome', 'DemaReal' ,'Gene', 'GeneIdea', 'CompBolsNaciEner', 'CompContEner'],axis=1)
df_consumption = df_consumption.rename(columns={'Timestamp': 'Timestamp2'})
df = pd.concat([df_production, df_consumption], axis=1)
df = df.fillna(0) 

# print(df.columns)
#FORECASTING PART
# df['Hour'] = pd.DatetimeIndex(df['Timestamp']).hour
# df['Month'] = pd.DatetimeIndex(df['Timestamp']).month
# df['Weekday'] = pd.DatetimeIndex(df['Timestamp']).weekday

# X = df[['Solar [kW]', 'Wind [kW]', 'Thermal [kW]', 'Cogeneration [kW]', 'Hydro [kW]','factorEmisionCO2e','Hour','Month','Weekday']]
X = df[['Solar [kW]', 'Wind [kW]', 'Thermal [kW]', 'Cogeneration [kW]', 'Hydro [kW]','factorEmisionCO2e']]
y = df['PrecBolsNaci']

# feature_names = ['Solar_kW', 'Wind_kW', 'Thermal_kW', 'Cogeneration_kW', 'Hydro_kW','factorEmisionCO2e','Hour','Month','Weekday']
feature_names = ['Solar_kW', 'Wind_kW', 'Thermal_kW', 'Cogeneration_kW', 'Hydro_kW','factorEmisionCO2e']
X.columns = feature_names
# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


results = pd.DataFrame(columns=['Method', 'Mean Squared Error', 'Coefficient of Determination', 'Mean_absolute_error'])

#Linear Regression
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
results = results.append({'Method': 'Linear Regression', 'Mean Squared Error': mean_squared_error(y_test, y_pred), 'Coefficient of Determination': r2_score(y_test, y_pred), 'Mean_absolute_error': np.mean(np.abs(y_pred - y_test))}, ignore_index=True)

#Polynomial Regression
poly = PolynomialFeatures(degree=4)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.fit_transform(X_test)
reg = LinearRegression()
reg.fit(X_train_poly, y_train)
y_pred = reg.predict(X_test_poly)

results = results.append({'Method': 'Polynomial Regression', 'Mean Squared Error': mean_squared_error(y_test, y_pred), 'Coefficient of Determination': r2_score(y_test, y_pred), 'Mean_absolute_error': np.mean(np.abs(y_pred - y_test))}, ignore_index=True)

#Random Forest Regression
reg = RandomForestRegressor(bootstrap=False, max_features=0.4, min_samples_leaf=1, min_samples_split=2, n_estimators=100)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
results = results.append({'Method': 'Random Forest Regression', 'Mean Squared Error': mean_squared_error(y_test, y_pred), 'Coefficient of Determination': r2_score(y_test, y_pred), 'Mean_absolute_error': np.mean(np.abs(y_pred - y_test))}, ignore_index=True)

#Gradient Boosting Regression
reg = GradientBoostingRegressor(random_state=42)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
results = results.append({'Method': 'Gradient Boosting Regression', 'Mean Squared Error': mean_squared_error(y_test, y_pred), 'Coefficient of Determination': r2_score(y_test, y_pred), 'Mean_absolute_error': np.mean(np.abs(y_pred - y_test))}, ignore_index=True)

#Lasso Regression
reg = Lasso(random_state=42)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
results = results.append({'Method': 'Lasso Regression', 'Mean Squared Error': mean_squared_error(y_test, y_pred), 'Coefficient of Determination': r2_score(y_test, y_pred), 'Mean_absolute_error': np.mean(np.abs(y_pred - y_test))}, ignore_index=True)

#plot each method in function of the mean squared error, coefficient of determination and mean absolute error
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].bar(results['Method'], results['Mean Squared Error'])
axs[0].set_title('Mean Squared Error')
axs[0].tick_params(axis='x', rotation=90)
axs[1].bar(results['Method'], results['Coefficient of Determination'])
axs[1].set_title('Coefficient of Determination')
axs[1].tick_params(axis='x', rotation=90)
axs[2].bar(results['Method'], results['Mean_absolute_error'])
axs[2].set_title('Mean Absolute Error')
axs[2].tick_params(axis='x', rotation=90)
plt.show()

