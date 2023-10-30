import pandas as pd
import datetime
import numpy as np
from pyomo.environ import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import itertools
import multiprocessing
import time
# print("test")
#hide warnings from pandas
pd.options.mode.chained_assignment = None  # default='warn'

start_time = datetime.datetime.now()
amount_of_days = 700
df_production = pd.read_csv("C:\\Users\\LIAT-ER\\Documents\\Thesis_Thibo\\Optimization_multithreading\\production_only_totals.csv")
df_consumption = pd.read_csv("C:\\Users\\LIAT-ER\\Documents\\Thesis_Thibo\\Optimization_multithreading\\consumption_varia.csv")
df_consumption = df_consumption.drop(columns=['Unnamed: 0','DemaCome', 'DemaReal' ,'Gene', 'GeneIdea', 'CompBolsNaciEner', 'CompContEner'],axis=1)
df_consumption = df_consumption.rename(columns={'Timestamp': 'Timestamp2'})
df = pd.concat([df_production, df_consumption], axis=1)
df = df.dropna() 
df = df.iloc[-24*amount_of_days:]
COP_USD = 0.00024
price_hydro = 66 #USD/MWh
price_hydro_list = [price_hydro*0.9, price_hydro*0.95, price_hydro, price_hydro*1.05, price_hydro*1.1]
price_solar = 68.41 #USD/MWh
price_solar_list = [price_solar*0.9, price_solar*0.95, price_solar, price_solar*1.05, price_solar*1.1]
price_wind = 90.97 #USD/MWh
price_wind_list = [price_wind*0.9, price_wind*0.95, price_wind, price_wind*1.05, price_wind*1.1]
price_thermal = 120 #USD/MWh
price_thermal_list = [price_thermal*0.9, price_thermal*0.95, price_thermal, price_thermal*1.05, price_thermal*1.1]
price_cogen = 67 #USD/MWh
price_cogen_list = [price_cogen*0.9, price_cogen*0.95, price_cogen, price_cogen*1.05, price_cogen*1.1]
co2_hydro = 24 #kg/MWh
co2_hydro_list = [co2_hydro*0.9, co2_hydro*0.95, co2_hydro, co2_hydro*1.05, co2_hydro*1.1]
co2_solar = 41 #kg/MWh
co2_solar_list = [co2_solar*0.9, co2_solar*0.95, co2_solar, co2_solar*1.05, co2_solar*1.1]
co2_wind = 11 #kg/MWh
co2_wind_list = [co2_wind*0.9, co2_wind*0.95, co2_wind, co2_wind*1.05, co2_wind*1.1]
co2_thermal = 550 #kg/MWh
co2_thermal_list = [co2_thermal*0.9, co2_thermal*0.95, co2_thermal, co2_thermal*1.05, co2_thermal*1.1]
co2_cogen = 217 #kg/MWh
co2_cogen_list = [co2_cogen*0.9, co2_cogen*0.95, co2_cogen, co2_cogen*1.05, co2_cogen*1.1]

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
coefficients = lregmodel.coef_
intercept = lregmodel.intercept_
df = df.iloc[-24*amount_of_days:]
old_cost = sum(df['Thermal [kW]'].iloc[1:]*price_thermal)+ sum(df['Solar [kW]'].iloc[1:]*price_solar)+sum(df['Wind [kW]'].iloc[1:]*price_wind)+sum(df['Cogeneration [kW]'].iloc[1:]*price_cogen)+sum(df['Hydro [kW]'].iloc[1:]*price_hydro)
old_co2 = sum(df['Total [kW]'].iloc[1:]*df['factorEmisionCO2e'].iloc[1:])
time_list = list(range(1, len(df['Total [kW]'].tolist())))
df['Solar DL [kW]'] = df['Solar [kW]'] / df['Solar [kW] max']
df['Wind DL [kW]'] = df['Wind [kW]'] / df['Wind [kW] max']
df = df.drop(columns=['month', 'year'],axis=1)
df = df.reset_index(drop=True)

df_sensitivity = pd.DataFrame(columns=['price_hydro [$/kWh]', 'price_solar [$/kWh]', 'price_wind [$/kWh]', 'price_thermal [$/kWh]', 'price_cogen [$/kWh]', 'co2_hydro [kg/kWh]', 'co2_solar [kg/kWh]', 'co2_wind [kg/kWh]', 'co2_thermal [kg/kWh]', 'co2_cogen [kg/kWh]', 'wind_capacity [kW]','solar_capacity [kW]','cost_conventional [$]', 'co2_conventional [kg]', 'co2_regression [kg]'])
#make csv file with these columns
df_sensitivity.to_csv('sensitivity_data.csv', index=False)

def sensitivity(combination):
    global combinations
    global df
    global df_sensitivity
    price_hydro = combination[0]
    price_solar = combination[1]
    price_wind = combination[2]
    price_thermal = combination[3]
    price_cogen = combination[4]
    co2_hydro = combination[5]
    co2_solar = combination[6]
    co2_wind = combination[7]
    co2_thermal = combination[8]
    co2_cogen = combination[9]

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
    model.E_hy_orig = Param(model.times, initialize=df['Hydro [kW]'].iloc[1:].to_dict())

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

    model.objective_co2_regression = Objective(expr=sum(((model.recalc_so[t]*model.coeff_LR_CO2_so) + 
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
    model.objective_co2_regression.deactivate()
    model.objective_co2_conventional.deactivate()
    results = SolverFactory('glpk').solve(model)
    # print('             ---- COST CONVENTIONAL----')
    # print('Desired solar capacity: ',model.solar_capacity.value/1000,'MW')
    # print('Desired wind capacity: ',model.wind_capacity.value/1000, 'MW')
    # time = datetime.datetime.now()
    # print('Time: ', time-start_time)
    # print('Old total cost: ',old_cost, 'USD')
    # print('New total cost: ',model.objective_cost(), 'USD')
    # print('Procentual difference: ',(old_cost-model.objective_cost())/old_cost*100,'%')

    #CO2 CONVENTIONAL
    model.objective_cost.deactivate()
    model.objective_co2_conventional.activate()
    results = SolverFactory('glpk').solve(model)

    # print('             ---- CO2 CONVENTIONAL ----')
    # print('Desired solar capacity: ',model.solar_capacity.value/1000,'MW')
    # print('Desired wind capacity: ',model.wind_capacity.value/1000, 'MW')

    # print('Old total CO2: ',old_co2, 'kg')
    # print('New total CO2: ',model.objective_co2_conventional(), 'kg')
    # print('Procentual difference: ',(old_co2-model.objective_co2_conventional())/old_co2*100,'%')

    if combination == combinations[0]: #only needs to be done once. The rest is the same
        # #CO2 WITH REGRESSION
        model.objective_co2_conventional.deactivate()
        model.objective_co2_regression.activate()
        results = SolverFactory('glpk').solve(model)

        # print('                 ---- CO2 WITH REGRESSION ----')
        # print('Desired solar capacity: ',model.solar_capacity.value/1000,'MW')
        # print('Desired wind capacity: ',model.wind_capacity.value/1000, 'MW')

        # print('Old total CO2: ',old_co2, 'kg')
        # print('New total CO2: ',model.objective_co2_regression(), 'kg')
        # print('Procentual difference: ',(old_co2-model.objective_co2_regression())/old_co2*100,'%')
        # df_sensitivity = df_sensitivity.append({'price_hydro [$/kWh]': price_hydro, 'price_solar [$/kWh]': price_solar, 'price_wind [$/kWh]': price_wind, 'price_thermal [$/kWh]': price_thermal, 'price_cogen [$/kWh]': price_cogen, 'co2_hydro [kg/kWh]': co2_hydro, 'co2_solar [kg/kWh]': co2_solar, 'co2_wind [kg/kWh]': co2_wind, 'co2_thermal [kg/kWh]': co2_thermal, 'co2_cogen [kg/kWh]': co2_cogen, 'wind_capacity [kW]': model.wind_capacity.value,'solar_capacity [kW]': model.solar_capacity.value,'cost_conventional [$]': model.objective_cost(), 'co2_conventional [kg]': model.objective_co2_conventional(), 'co2_regression [kg]': model.objective_co2_regression()}, ignore_index=True)
        data = {
            'price_hydro [$/kWh]': price_hydro,
            'price_solar [$/kWh]': price_solar,
            'price_wind [$/kWh]': price_wind,
            'price_thermal [$/kWh]': price_thermal,
            'price_cogen [$/kWh]': price_cogen,
            'co2_hydro [kg/kWh]': co2_hydro,
            'co2_solar [kg/kWh]': co2_solar,
            'co2_wind [kg/kWh]': co2_wind,
            'co2_thermal [kg/kWh]': co2_thermal,
            'co2_cogen [kg/kWh]': co2_cogen,
            'wind_capacity [kW]': model.wind_capacity.value,
            'solar_capacity [kW]': model.solar_capacity.value,
            'cost_conventional [$]': model.objective_cost(),
            'co2_conventional [kg]': model.objective_co2_conventional(),
            'co2_regression [kg]': model.objective_co2_regression()}
        new_row = pd.DataFrame(data, index=[0])
    else:
        data = {
            'price_hydro [$/kWh]': price_hydro,
            'price_solar [$/kWh]': price_solar,
            'price_wind [$/kWh]': price_wind,
            'price_thermal [$/kWh]': price_thermal,
            'price_cogen [$/kWh]': price_cogen,
            'co2_hydro [kg/kWh]': co2_hydro,
            'co2_solar [kg/kWh]': co2_solar,
            'co2_wind [kg/kWh]': co2_wind,
            'co2_thermal [kg/kWh]': co2_thermal,
            'co2_cogen [kg/kWh]': co2_cogen,
            'wind_capacity [kW]': model.wind_capacity.value,
            'solar_capacity [kW]': model.solar_capacity.value,
            'cost_conventional [$]': model.objective_cost(),
            'co2_conventional [kg]': model.objective_co2_conventional(),
            'co2_regression [kg]': 0}
        new_row = pd.DataFrame(data, index=[0])
    df_sensitivity = pd.concat([df_sensitivity, new_row], ignore_index=True)
    #export the last row including header to a csv file
    df_sensitivity.tail(1).to_csv('sensitivity_data.csv', mode='a', header=False, index=False)
    time.sleep(1)
    print(df_sensitivity.shape)


price_lists = [price_hydro_list, price_solar_list, price_wind_list, price_thermal_list, price_cogen_list]
co2_lists = [co2_hydro_list, co2_solar_list, co2_wind_list, co2_thermal_list, co2_cogen_list]
combinations = list(itertools.product(*price_lists, *co2_lists))

def main():
    global combinations, df, df_sensitivity  # Access global variables
    num_processes = 2
    if __name__ == '__main__':
        multiprocessing.freeze_support() 
        with multiprocessing.Pool(num_processes) as pool:
            pool.map(sensitivity, combinations)


if __name__ == '__main__':
    main() 