import pandas as pd
import datetime
import numpy as np
from pyomo.environ import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

amount_of_days = 700
df_production = pd.read_csv("C:\\Users\\thibo\\OneDrive - Hanzehogeschool Groningen\\Documenten\\Github\\UNAL_thesis\\xm_API\\production\\production_only_totals.csv")
df_consumption = pd.read_csv("C:\\Users\\thibo\\OneDrive - Hanzehogeschool Groningen\\Documenten\\Github\\UNAL_thesis\\xm_API\\consumption\\consumption_varia.csv")
df_consumption = df_consumption.drop(columns=['Unnamed: 0','DemaCome', 'DemaReal' ,'Gene', 'GeneIdea', 'CompBolsNaciEner', 'CompContEner'],axis=1)
df_consumption = df_consumption.rename(columns={'Timestamp': 'Timestamp2'})
df = pd.concat([df_production, df_consumption], axis=1)
df = df.dropna() 
df = df.iloc[-24*amount_of_days:]
COP_USD = 0.00024
price_hydro = 66 #USD/MWh
price_solar = 68.41 #USD/MWh
price_wind = 90.97 #USD/MWh
price_thermal = 120 #USD/MWh
price_cogen = 67 #USD/MWh 
co2_hydro = 24 #kg/MWh
co2_solar = 41 #kg/MWh
co2_wind = 11 #kg/MWh
co2_thermal = 550 #kg/MWh
co2_cogen = 217 #kg/MWh

columns_to_determine_monthly_max = ['Cogeneration [kW]', 'Hydro [kW]', 'Solar [kW]', 'Wind [kW]', 'Thermal [kW]']
df['month'] = pd.DatetimeIndex(df['Timestamp']).month
df['year'] = pd.DatetimeIndex(df['Timestamp']).year
for column in columns_to_determine_monthly_max:
    df[column + ' max'] = df.groupby(['month', 'year'])[column].transform(max)

#FORECASTING CO2
X = df[['Solar [kW]', 'Wind [kW]', 'Thermal [kW]', 'Cogeneration [kW]', 'Hydro [kW]']]  #CO2
y = df['factorEmisionCO2e'] #CO2
feature_names = ['Solar_kW', 'Wind_kW', 'Thermal_kW', 'Cogeneration_kW', 'Hydro_kW']
X.columns = feature_names
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lregmodel = LinearRegression()
lregmodel.fit(X_train, y_train)
coefficients = lregmodel.coef_  # Slope(s)
# print(coefficients.tolist())
intercept = lregmodel.intercept_  # Intercept


plt.plot(pd.to_datetime(df['Timestamp']), df['Cogeneration [kW]'],label='Cogeneration [kW]', color='gray')
plt.plot(pd.to_datetime(df['Timestamp']),df['Cogeneration [kW] max'], label='Cogeneration [kW] max', color='black')
plt.xlabel('Time')
plt.xticks(rotation=45)
plt.ylabel('Power (kW)')
plt.title('Cogeneration: installed capacity')
plt.legend()
plt.show()

plt.plot(pd.to_datetime(df['Timestamp']), df['Hydro [kW]'], label='Hydro [kW]', color = 'darkblue')
plt.plot(pd.to_datetime(df['Timestamp']),df['Hydro [kW] max'], label='Hydro [kW] max', color='black')
plt.xlabel('Time')
plt.xticks(rotation=45)
plt.ylabel('Power (kW)')
plt.title('Hydro: installed capacity')
plt.legend()
plt.show()

plt.plot(pd.to_datetime(df['Timestamp']), df['Solar [kW]'], label='Solar [kW]', color = 'yellow')
plt.plot(pd.to_datetime(df['Timestamp']),df['Solar [kW] max'], label='Solar [kW] max', color='black')
plt.xlabel('Time')
plt.xticks(rotation=45)
plt.ylabel('Power (kW)')
plt.title('Solar: installed capacity')
plt.legend()
plt.show()

plt.plot(pd.to_datetime(df['Timestamp']), df['Wind [kW]'], label='Wind [kW]', color = 'blue')
plt.plot(pd.to_datetime(df['Timestamp']),df['Wind [kW] max'], label='Wind [kW] max', color='black')
plt.xlabel('Time')
plt.xticks(rotation=45)
plt.ylabel('Power (kW)')
plt.title('Wind: installed capacity')
plt.legend()
plt.show()

plt.plot(pd.to_datetime(df['Timestamp']), df['Thermal [kW]'], label='Thermal [kW]', color = 'red')
plt.plot(pd.to_datetime(df['Timestamp']),df['Thermal [kW] max'], label='Thermal [kW] max', color='black')
plt.xlabel('Time')
plt.xticks(rotation=45)
plt.ylabel('Power (kW)')
plt.title('Thermal: installed capacity')
plt.legend()
plt.show()


# df = df.iloc[-24*amount_of_days:]
#OMZETTEN NAAR DOLLARS
old_cost = sum(df['Thermal [kW]'].iloc[1:]*price_thermal)+ sum(df['Solar [kW]'].iloc[1:]*price_solar)+sum(df['Wind [kW]'].iloc[1:]*price_wind)+sum(df['Cogeneration [kW]'].iloc[1:]*price_cogen)+sum(df['Hydro [kW]'].iloc[1:]*price_hydro)
old_co2 = sum(df['Total [kW]'].iloc[1:]*df['factorEmisionCO2e'].iloc[1:])
time_list = list(range(1, len(df['Total [kW]'].tolist())))
df['Solar DL [kW]'] = df['Solar [kW]'] / df['Solar [kW] max']
df['Wind DL [kW]'] = df['Wind [kW]'] / df['Wind [kW] max']
df = df.drop(columns=['month', 'year'],axis=1)

df = df.reset_index(drop=True)

model = ConcreteModel()
model.times = Set(initialize=time_list)

model.solar_capacity = Var(within=NonNegativeReals, bounds = (0,32000000)) #max capacity to determine of sources https://www.sei.org/publications/solar-wind-power-colombia-2022/
model.wind_capacity = Var(within=NonNegativeReals,  bounds=(0,30000000)) #https://www.sei.org/publications/solar-wind-power-colombia-2022/

model.production_co = Var(model.times, within = NonNegativeReals)
model.production_hy = Var(model.times, within = NonNegativeReals)

model.production_th = Var(model.times, within = NonNegativeReals)

model.recalc_wi = Var(model.times, within = NonNegativeReals)
model.recalc_so = Var(model.times, within = NonNegativeReals)

model.epsilon=Param(initialize=10000,mutable=True) #nodig voor pareto
model.E_tot = Param(model.times, initialize=df['Total [kW]'].iloc[1:].to_dict())
model.E_wi_partial = Param(model.times, initialize=df['Wind DL [kW]'].iloc[1:].to_dict())
model.E_so_partial = Param(model.times, initialize=df['Solar DL [kW]'].iloc[1:].to_dict())
model.E_co_max = Param(model.times, initialize=df['Cogeneration [kW] max'].iloc[1:].to_dict())
model.E_hy_max = Param(model.times, initialize=df['Hydro [kW] max'].iloc[1:].to_dict())
model.E_th_max = Param(model.times, initialize=df['Thermal [kW] max'].iloc[1:].to_dict())
# model.orig_production_co = Param(model.times, initialize=df['Cogeneration [kW]'].iloc[1:].to_dict())
model.E_hy_orig = Param(model.times, initialize=df['Hydro [kW]'].iloc[1:].to_dict())
# model.orig_production_th = Param(model.times, initialize=df['Thermal [kW]'].iloc[1:].to_dict())
# model.coefficients_LR_CO2 = Param(initialize= {'coefficients': coefficients.tolist()} )


model.price_th = Param(initialize=price_thermal)
model.price_co = Param(initialize=price_cogen)
model.price_hy = Param(initialize=price_hydro)
model.price_so = Param(initialize=price_solar)
model.price_wi = Param(initialize=price_wind)
model.CO2_so = Param(initialize=co2_solar)
model.CO2_wi = Param(initialize=co2_wind)
model.CO2_th = Param(initialize=co2_thermal)
model.CO2_co = Param(initialize=co2_cogen)
model.CO2_hy = Param(initialize=co2_hydro)
# 'Solar [kW]', 'Wind [kW]', 'Thermal [kW]', 'Cogeneration [kW]', 'Hydro [kW]'
model.coeff_LR_CO2_so = Param(initialize=coefficients.tolist()[0])
model.coeff_LR_CO2_wi = Param(initialize=coefficients.tolist()[1])
model.coeff_LR_CO2_th = Param(initialize=coefficients.tolist()[2])
model.coeff_LR_CO2_co = Param(initialize=coefficients.tolist()[3])
model.coeff_LR_CO2_hy = Param(initialize=coefficients.tolist()[4])
model.intercept_LR_CO2 = Param(initialize=intercept)

def recalc_wi_rule(model, t):
    return model.recalc_wi[t] == model.E_wi_partial[t]*model.wind_capacity
model.recalc_wi_constraint = Constraint(model.times, rule=recalc_wi_rule)

def recalc_so_rule(model, t):
    return model.recalc_so[t] == model.E_so_partial[t]*model.solar_capacity
model.recalc_so_constraint = Constraint(model.times, rule=recalc_so_rule)

def demand_rule(model, t):
    return model.production_th[t] + model.production_co[t] + model.production_hy[t] + model.recalc_so[t] + model.recalc_wi[t] == model.E_tot[t]
model.demand_constraint = Constraint(model.times, rule=demand_rule)

def max_constraint_co(model, t):
    return model.production_co[t] <= model.E_co_max[t]
model.max_constraint_co = Constraint(model.times, rule=max_constraint_co)

def max_constraint_th(model, t):
    return model.production_th[t] <= model.E_th_max[t]
model.max_constraint_th = Constraint(model.times, rule=max_constraint_th)

def constraint_hy(model, t): #houdt rekening dat deze zeker niet meer kan produceren dan origineel
    return model.production_hy[t] <= model.E_hy_orig[t]
model.max_constraint_hy = Constraint(model.times, rule=constraint_hy)




model.objective_cost = Objective(expr=sum((model.production_th[t]*model.price_th) + 
                                     (model.production_co[t]*model.price_co) + 
                                     (model.production_hy[t]*model.price_hy) +
                                     (model.recalc_wi[t]*model.price_wi) +
                                     (model.recalc_so[t]*model.price_so) for t in model.times), sense=minimize)

#mogelijks deze koppelen aan de targets van de CO2
model.objective_co2 = Objective(expr=sum(((model.recalc_so[t]*model.coeff_LR_CO2_so) + 
                                         (model.recalc_wi[t]*model.coeff_LR_CO2_wi) + 
                                         (model.production_th[t]*model.coeff_LR_CO2_th) +
                                         (model.production_co[t]*model.coeff_LR_CO2_co) +
                                         (model.production_hy[t]*model.coeff_LR_CO2_hy) + model.intercept_LR_CO2)*model.E_tot[t] for t in model.times), sense=minimize)
model.objective_co2_conventional = Objective(expr=sum(((model.recalc_so[t]*model.CO2_so) +
                                            (model.recalc_wi[t]*model.CO2_wi) +
                                            (model.production_th[t]*model.CO2_th) +
                                            (model.production_co[t]*model.CO2_co) +
                                            (model.production_hy[t]*model.CO2_hy)) for t in model.times), sense=minimize)

#Individuele optimalisatie
#COST CONVENTIONAL
model.objective_co2.deactivate()
model.objective_co2_conventional.deactivate()
results = SolverFactory('ipopt').solve(model)
print('---- COST CONVENTIONAL----')
print('Desired solar capacity: ',model.solar_capacity.value/1000,'MW')
print('Desired wind capacity: ',model.wind_capacity.value/1000, 'MW')

print('Old total cost: ',old_cost, 'USD')
print('New total cost: ',model.objective_cost(), 'USD')
print('Procentual difference: ',(old_cost-model.objective_cost())/old_cost*100,'%')

#CO2 CONVENTIONAL
model.objective_cost.deactivate()
model.objective_co2_conventional.activate()
results = SolverFactory('ipopt').solve(model)

print('---- CO2 CONVENTIONAL ----')
print('Desired solar capacity: ',model.solar_capacity.value/1000,'MW')
print('Desired wind capacity: ',model.wind_capacity.value/1000, 'MW')

print('Old total CO2: ',old_co2, 'kg')
print('New total CO2: ',model.objective_co2_conventional(), 'kg')
print('Procentual difference: ',(old_co2-model.objective_co2_conventional())/old_co2*100,'%')


#CO2 WITH REGRESSION
model.objective_co2_conventional.deactivate()
model.objective_co2.activate()
results = SolverFactory('ipopt').solve(model)

print('---- CO2 WITH REGRESSION ----')
print('Desired solar capacity: ',model.solar_capacity.value/1000,'MW')
print('Desired wind capacity: ',model.wind_capacity.value/1000, 'MW')

print('Old total CO2: ',old_co2, 'kg')
print('New total CO2: ',model.objective_co2(), 'kg')
print('Procentual difference: ',(old_co2-model.objective_co2())/old_co2*100,'%')
