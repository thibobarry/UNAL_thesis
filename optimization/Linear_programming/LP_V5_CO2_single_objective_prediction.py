import pandas as pd
import datetime
from pyomo.environ import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import time
COP_USD = 4217
#KIJKEN OF RFR WERKT BINNEN LP
#MSS OOK LREG AANPASSEN ZODAT HET ZELFDE STIJL HEEFT ALS RFR
#NOG VOLLEDIG TESTEN

pd.options.mode.chained_assignment = None  # default='warn'

start_time = datetime.datetime.now()
amount_of_days = 700
# df_production = pd.read_csv("C:\\Users\\LIAT-ER\\Documents\\Thesis_Thibo\\Optimization_multithreading\\production_only_totals.csv")
df_production = pd.read_csv("C:\\Users\\thibo\\OneDrive - Hanzehogeschool Groningen\\Documenten\\Github\\UNAL_thesis\\xm_API\\production\\production_only_totals.csv")
# df_consumption = pd.read_csv("C:\\Users\\LIAT-ER\\Documents\\Thesis_Thibo\\Optimization_multithreading\\consumption_varia.csv")
df_consumption = pd.read_csv("C:\\Users\\thibo\\OneDrive - Hanzehogeschool Groningen\\Documenten\\Github\\UNAL_thesis\\xm_API\\consumption\\consumption_varia.csv")
# print(df_consumption.columns)
df_consumption = df_consumption.drop(columns=['Unnamed: 0','DemaCome', 'DemaReal' ,'Gene', 'GeneIdea', 'CompContEner'],axis=1)
df_consumption = df_consumption.rename(columns={'Timestamp': 'Timestamp2'})
df = pd.concat([df_production, df_consumption], axis=1)
df = df.dropna() 
df = df.iloc[-24*amount_of_days:]

columns_to_determine_monthly_max = ['Cogeneration [kW]', 'Hydro [kW]', 'Solar [kW]', 'Wind [kW]', 'Thermal [kW]']
df['month'] = pd.DatetimeIndex(df['Timestamp']).month
df['year'] = pd.DatetimeIndex(df['Timestamp']).year
for column in columns_to_determine_monthly_max:
    df[column + ' max'] = df.groupby(['month', 'year'])[column].transform(max)

# FORECASTING CO2
X = df[['Solar [kW]', 'Wind [kW]', 'Thermal [kW]', 'Cogeneration [kW]', 'Hydro [kW]']]  #CO2
y = df['factorEmisionCO2e'] #CO2
feature_names = ['Solar_kW', 'Wind_kW', 'Thermal_kW', 'Cogeneration_kW', 'Hydro_kW']
X.columns = feature_names
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lregmodel = LinearRegression()
lregmodel.fit(X_train, y_train)
coefficients = lregmodel.coef_
intercept = lregmodel.intercept_

#forecasting cost
# X = df[['Solar [kW]', 'Wind [kW]', 'Thermal [kW]', 'Cogeneration [kW]', 'Hydro [kW]']]  #cost
# y = df['CompBolsNaciEner'] #cost
# feature_names = ['Solar_kW', 'Wind_kW', 'Thermal_kW', 'Cogeneration_kW', 'Hydro_kW']
# X.columns = feature_names
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# rfr = RandomForestRegressor(bootstrap=False, max_features=0.4, min_samples_leaf=1, min_samples_split=2, n_estimators=100)
# rfr.fit(X_train, y_train)

df = df.iloc[-24*amount_of_days:]
old_cost = sum(df['Total [kW]'].iloc[1:]*df['CompBolsNaciEner'].iloc[1:])/COP_USD #HIER NOG PROBLEEMPJE???????
old_cost_2 = (sum(df['Solar [kW]'])*68/1000 + 
              sum(df['Wind [kW]'])*91/1000 +
                sum(df['Thermal [kW]'])*120/1000 +
                sum(df['Cogeneration [kW]'])*67/1000 +
                sum(df['Hydro [kW]'])*66/1000)
old_co2 = sum(df['Total [kW]'].iloc[1:]*df['factorEmisionCO2e'].iloc[1:])
old_co2_2 = (sum(df['Solar [kW]'])*0.41 +
            sum(df['Wind [kW]'])*0.11 +
            sum(df['Thermal [kW]'])*5.5 +
            sum(df['Cogeneration [kW]'])*2.17 +
            sum(df['Hydro [kW]'])*0.24)

time_list = list(range(1, len(df['Total [kW]'].tolist())))
df['Solar DL [kW]'] = df['Solar [kW]'] / df['Solar [kW] max']
df['Wind DL [kW]'] = df['Wind [kW]'] / df['Wind [kW] max']
df = df.drop(columns=['month', 'year','CompBolsNaciEner'],axis=1)
df = df.reset_index(drop=True)

df_sensitivity = pd.DataFrame(columns=['strategy', 'wind_capacity [kW]','solar_capacity [kW]','cost_conventional [$]', 'co2_regression [kg]','original_cost [$]', 'original_co2 [kg]'])
df_sensitivity.to_csv('prediction_optimization.csv', index=False)

model = ConcreteModel()
model.times = Set(initialize=time_list)

model.solar_capacity = Var(within=NonNegativeReals, bounds = (0,32000000)) #max capacity to determine of sources https://www.sei.org/publications/solar-wind-power-colombia-2022/
model.wind_capacity = Var(within=NonNegativeReals,  bounds=(0,30000000)) #https://www.sei.org/publications/solar-wind-power-colombia-2022/

model.E_co = Var(model.times, within = NonNegativeReals)
model.E_hy = Var(model.times, within = NonNegativeReals)

model.E_th = Var(model.times, within = NonNegativeReals)

model.recalc_wi = Var(model.times, within = NonNegativeReals)
model.recalc_so = Var(model.times, within = NonNegativeReals)

model.new_cost_non_primary = Var(model.times, within = NonNegativeReals)

model.epsilon=Param(initialize=10000,mutable=True) #nodig voor pareto
model.E_tot = Param(model.times, initialize=df['Total [kW]'].iloc[1:].to_dict())
model.E_wi_partial = Param(model.times, initialize=df['Wind DL [kW]'].iloc[1:].to_dict())
model.E_so_partial = Param(model.times, initialize=df['Solar DL [kW]'].iloc[1:].to_dict())
model.E_so_orig = Param(model.times, initialize=df['Solar [kW]'].iloc[1:].to_dict()) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
model.E_wi_orig = Param(model.times, initialize=df['Wind [kW]'].iloc[1:].to_dict()) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
model.E_th_orig = Param(model.times, initialize=df['Thermal [kW]'].iloc[1:].to_dict()) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
model.E_co_orig = Param(model.times, initialize=df['Cogeneration [kW]'].iloc[1:].to_dict()) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
model.E_hy_orig = Param(model.times, initialize=df['Hydro [kW]'].iloc[1:].to_dict())
model.E_co_max = Param(model.times, initialize=df['Cogeneration [kW] max'].iloc[1:].to_dict())
model.E_hy_max = Param(model.times, initialize=df['Hydro [kW] max'].iloc[1:].to_dict())
model.E_th_max = Param(model.times, initialize=df['Thermal [kW] max'].iloc[1:].to_dict())

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
    return model.E_th[t] + model.E_co[t] + model.E_hy[t] + model.recalc_so[t] + model.recalc_wi[t] == model.E_tot[t]
model.demand_constraint = Constraint(model.times, rule=demand_rule)

def max_constraint_co(model, t):
    return model.E_co[t] <= model.E_co_max[t]
model.max_constraint_co = Constraint(model.times, rule=max_constraint_co)

def max_constraint_th(model, t):
    return model.E_th[t] <= model.E_th_max[t]
model.max_constraint_th = Constraint(model.times, rule=max_constraint_th)

def constraint_hy(model, t):
    return model.E_hy[t] <= model.E_hy_orig[t]
model.max_constraint_hy = Constraint(model.times, rule=constraint_hy)

def new_cost_non_primary(model,t): #NOG TOEVOEGEN AAN THESIS
    return model.new_cost_non_primary[t] ==((model.recalc_so[t]*68/1000) +
                                        (model.recalc_wi[t]*91/1000) +
                                        (model.E_th[t]*120/1000) +
                                        (model.E_co[t]*67/1000) +
                                        (model.E_hy[t]*66/1000))
    # totaal aantal dollars want prijs was in dollar per MWh
model.constraint_new_cost_non_primary = Constraint(model.times, rule=new_cost_non_primary)


model.objective_co2_regression = Objective(expr=sum(((model.recalc_so[t]*model.coeff_LR_CO2_so) + 
                                            (model.recalc_wi[t]*model.coeff_LR_CO2_wi) + 
                                            (model.E_th[t]*model.coeff_LR_CO2_th) +
                                            (model.E_co[t]*model.coeff_LR_CO2_co) +
                                            (model.E_hy[t]*model.coeff_LR_CO2_hy) + model.intercept_LR_CO2)*model.E_tot[t] for t in model.times), sense=minimize)


#CO2 WITH REGRESSION
# model.objective_cost_rfr.deactivate()
model.objective_co2_regression.activate()
results = SolverFactory('glpk').solve(model)
new_cost = [] #om de cost te berekenen als side effect na de optimizatie van co2
for t in model.times:
    new_cost.append(model.new_cost_non_primary[t].value) #originele value is in $
# 'strategy', 'wind_capacity [kW]','solar_capacity [kW]','cost_conventional [$]', 'co2_regression [kg]','original_cost [$]', 'original_co2 [kg]'
data = {
    'strategy': 'co2_regression',
    'wind_capacity [kW]': model.wind_capacity.value,
    'solar_capacity [kW]': model.solar_capacity.value,
    'cost_conventional [$]': sum(new_cost),
    'co2_regression [kg]': model.objective_co2_regression()/1000,
    'original_cost [$]': old_cost_2,
    'original_co2 [kg]': old_co2/1000  #originele eenheid is g/kWh
}
new_row = pd.DataFrame(data, index=[0])
df_sensitivity = pd.concat([df_sensitivity, new_row], ignore_index=True)
df_sensitivity.to_csv('prediction_optimization.csv', mode='a', header=False, index=False)
time.sleep(1)
print(df_sensitivity.shape)
