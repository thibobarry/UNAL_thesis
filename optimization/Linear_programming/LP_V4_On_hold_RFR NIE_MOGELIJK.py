import pandas as pd
import datetime
import numpy as np
from pyomo.environ import *
from pyomo.environ import value
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

amount_of_days = 200
df_production = pd.read_csv("C:\\Users\\thibo\\OneDrive - Hanzehogeschool Groningen\\Documenten\\Github\\UNAL_thesis\\xm_API\\production\\production_only_totals.csv")
df_consumption = pd.read_csv("C:\\Users\\thibo\\OneDrive - Hanzehogeschool Groningen\\Documenten\\Github\\UNAL_thesis\\xm_API\\consumption\\consumption_varia.csv")
df_consumption = df_consumption.drop(columns=['Unnamed: 0','DemaCome', 'DemaReal' ,'Gene', 'GeneIdea', 'CompBolsNaciEner', 'CompContEner'],axis=1)
df_consumption = df_consumption.rename(columns={'Timestamp': 'Timestamp2'})
df = pd.concat([df_production, df_consumption], axis=1)
df = df.fillna(0) 
#FORECASTING PART
X = df[['Solar [kW]', 'Wind [kW]', 'Thermal [kW]', 'Cogeneration [kW]', 'Hydro [kW]']]
y = df['PrecBolsNaci']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, random_state=42)
rfr = RandomForestRegressor(bootstrap=False, max_features=0.4, min_samples_leaf=1, min_samples_split=2, n_estimators=100)
rfr.fit(X_train, y_train)

y_pred = rfr.predict(X_test)
print(X_test)
print(y_pred[0])
print(y_pred)
# # print('Mean squared error: ',mean_squared_error(y_test, y_pred))
# # print('Mean absolute error: ',mean_absolute_error(y_test, y_pred))
# # print('R2 score: ',rfr.score(X_test, y_test))
columns_to_determine_monthly_max = ['Cogeneration [kW]', 'Hydro [kW]', 'Solar [kW]', 'Wind [kW]', 'Thermal [kW]']
df['month'] = pd.DatetimeIndex(df['Timestamp']).month
df['year'] = pd.DatetimeIndex(df['Timestamp']).year
for column in columns_to_determine_monthly_max:
    df[column + ' max'] = df.groupby(['month', 'year'])[column].transform(max)

# #plot Cogeneration [kW] and Cogeneration [kW] max to see if it works
# # plt.plot(df['Cogeneration [kW]'], label='Cogeneration [kW]')
# # plt.plot(df['Cogeneration [kW] max'], label='Cogeneration [kW] max')
# # plt.legend()
# # plt.show()

# # plt.plot(df['Hydro [kW]'], label='Hydro [kW]')
# # plt.plot(df['Hydro [kW] max'], label='Hydro [kW] max')
# # plt.legend()
# # plt.show()

# # plt.plot(df['Solar [kW]'], label='Solar [kW]')
# # plt.plot(df['Solar [kW] max'], label='Solar [kW] max')
# # plt.legend()
# # plt.show()

# # plt.plot(df['Wind [kW]'], label='Wind [kW]')
# # plt.plot(df['Wind [kW] max'], label='Wind [kW] max')
# # plt.legend()
# # plt.show()

# # plt.plot(df['Thermal [kW]'], label='Thermal [kW]')
# # plt.plot(df['Thermal [kW] max'], label='Thermal [kW] max')
# # plt.legend()
# # plt.show()

# #uitbreidingen
# #1) die maximale capaciteit per maand meegeven wind en zon (check zie grafieken boven)
# #2) constraints toevoegen waardoor alle andere units niet hoger dan hun max kunnen gaan (check)
# #3) voeg toe dat de totale productie van een nieuwe maand niet hoger kan zijn dan de originele totale productie van hydro



df = df.iloc[-24*amount_of_days:]
time_list = list(range(1, len(df['Total [kW]'].tolist())))
df['Solar DL [kW]'] = df['Solar [kW]'] / df['Solar [kW] max']
df['Wind DL [kW]'] = df['Wind [kW]'] / df['Wind [kW] max']
df = df.drop(columns=['month', 'year'],axis=1)
# print(df.head(10))

df = df.reset_index(drop=True)

model = ConcreteModel()
model.times = Set(initialize=time_list)

model.solar_capacity = Var(within=NonNegativeReals, bounds = (0,32000000)) #max capacity to determine of sources https://www.sei.org/publications/solar-wind-power-colombia-2022/
model.wind_capacity = Var(within=NonNegativeReals,  bounds=(0,30000000)) #https://www.sei.org/publications/solar-wind-power-colombia-2022/

model.production_co = Var(model.times, within = NonNegativeReals)
model.production_hy = Var(model.times, within = NonNegativeReals)
# model.production_so = Var(model.times, within = NonNegativeReals)
# model.production_wi = Var(model.times, within = NonNegativeReals)
model.production_th = Var(model.times, within = NonNegativeReals)
model.predicted_price = Var(model.times, within = NonNegativeReals)

model.recalc_wi = Var(model.times, within = NonNegativeReals)
model.recalc_so = Var(model.times, within = NonNegativeReals)


model.production_total = Param(model.times, initialize=df['Total [kW]'].iloc[1:].to_dict())
model.production_wi_partial_load = Param(model.times, initialize=df['Wind DL [kW]'].iloc[1:].to_dict())
model.production_so_partial_load = Param(model.times, initialize=df['Solar DL [kW]'].iloc[1:].to_dict())
model.max_production_co = Param(model.times, initialize=df['Cogeneration [kW] max'].iloc[1:].to_dict())
model.max_production_hy = Param(model.times, initialize=df['Hydro [kW] max'].iloc[1:].to_dict())
model.max_production_th = Param(model.times, initialize=df['Thermal [kW] max'].iloc[1:].to_dict())
model.orig_production_co = Param(model.times, initialize=df['Cogeneration [kW]'].iloc[1:].to_dict())
model.orig_production_hy = Param(model.times, initialize=df['Hydro [kW]'].iloc[1:].to_dict())
model.orig_production_th = Param(model.times, initialize=df['Thermal [kW]'].iloc[1:].to_dict())
# model.price_th = Param(initialize=price_thermal)
# model.price_co = Param(initialize=price_cogen)
# model.price_hy = Param(initialize=price_hydro)
# model.price_so = Param(initialize=price_solar)
# model.price_wi = Param(initialize=price_wind)

def recalc_wi_rule(model, t):
    return model.recalc_wi[t] == model.production_wi_partial_load[t]*model.wind_capacity
model.recalc_wi_constraint = Constraint(model.times, rule=recalc_wi_rule)

def recalc_so_rule(model, t):
    return model.recalc_so[t] == model.production_so_partial_load[t]*model.solar_capacity
model.recalc_so_constraint = Constraint(model.times, rule=recalc_so_rule)

def demand_rule(model, t):
    return model.production_th[t] + model.production_co[t] + model.production_hy[t] + model.recalc_so[t] + model.recalc_wi[t] == model.production_total[t]
model.demand_constraint = Constraint(model.times, rule=demand_rule)

def max_constraint_co(model, t):
    return model.production_co[t] <= model.max_production_co[t]
model.max_constraint_co = Constraint(model.times, rule=max_constraint_co)

def max_constraint_th(model, t):
    return model.production_th[t] <= model.max_production_th[t]
model.max_constraint_th = Constraint(model.times, rule=max_constraint_th)

def constraint_hy(model, t): #houdt rekening dat deze zeker niet meer kan produceren dan origineel
    return model.production_hy[t] <= model.orig_production_hy[t]
model.max_constraint_hy = Constraint(model.times, rule=constraint_hy)

#kijken of deze constraint kan
def price_constraint(model, t):
    input_data = [[model.recalc_so[t], model.recalc_wi[t], model.production_th[t], model.production_co[t], model.production_hy[t]]]
    return   model.predicted_price[t] == rfr.predict(input_data)[0]
model.price_constraint = Constraint(model.times, rule=price_constraint)

model.objective = Objective(expr=sum(model.predicted_price[t]* model.production_total[t] for t in model.times), sense=minimize)

solver = SolverFactory('glpk')
results = solver.solve(model)

# [['Solar [kW]', 'Wind [kW]', 'Thermal [kW]', 'Cogeneration [kW]', 'Hydro [kW]']]

print('---------')
print('Desired solar capacity: ',model.solar_capacity.value/1000,'MW')
print('Desired wind capacity: ',model.wind_capacity.value, 'MW')
#print objective function value
print('---------')
old_cost = sum(df['Total [kW]'].iloc[1:].tolist())*(df['PrecBolsNaci'].iloc[1:].tolist())
print('Old total cost: ',old_cost, 'USD')
print('New total cost: ',model.objective(), 'USD')
print('Procentual difference: ',(old_cost-model.objective())/old_cost*100,'%')