import pandas as pd
import datetime
from pyomo.environ import *
import time

# print("test")
#hide warnings from pandas
pd.options.mode.chained_assignment = None  # default='warn'

start_time = datetime.datetime.now()
amount_of_days = 700
# df_production = pd.read_csv("C:\\Users\\LIAT-ER\\Documents\\Thesis_Thibo\\Optimization_multithreading\\production_only_totals.csv")
df_production = pd.read_csv("C:\\Users\\thibo\\OneDrive - Hanzehogeschool Groningen\\Documenten\\Github\\UNAL_thesis\\xm_API\\production\\production_only_totals.csv")
# df_consumption = pd.read_csv("C:\\Users\\LIAT-ER\\Documents\\Thesis_Thibo\\Optimization_multithreading\\consumption_varia.csv")
df_consumption = pd.read_csv("C:\\Users\\thibo\\OneDrive - Hanzehogeschool Groningen\\Documenten\\Github\\UNAL_thesis\\xm_API\\consumption\\consumption_varia.csv")
df_consumption = df_consumption.drop(columns=['Unnamed: 0','DemaCome', 'DemaReal' ,'Gene', 'GeneIdea', 'CompBolsNaciEner', 'CompContEner'],axis=1)
df_consumption = df_consumption.rename(columns={'Timestamp': 'Timestamp2'})
df = pd.concat([df_production, df_consumption], axis=1)
df = df.dropna() 
df = df.iloc[-24*amount_of_days:]
price_hydro = 66/1000 #USD/kWh
price_solar = 68.41/1000 #USD/kWh
price_wind = 90.97/1000 #USD/kWh
price_thermal = 120/1000 #USD/kWh
price_cogen = 67/1000 #USD/kWh
co2_hydro = 24/1000 #kg/kWh
co2_solar = 41/1000 #kg/kWh
co2_wind = 11/1000 #kg/kWh
co2_thermal = 550/1000 #kg/kWh
co2_cogen = 217/1000 #kg/kWh


columns_to_determine_monthly_max = ['Cogeneration [kW]', 'Hydro [kW]', 'Solar [kW]', 'Wind [kW]', 'Thermal [kW]']
df['month'] = pd.DatetimeIndex(df['Timestamp']).month
df['year'] = pd.DatetimeIndex(df['Timestamp']).year
for column in columns_to_determine_monthly_max:
    df[column + ' max'] = df.groupby(['month', 'year'])[column].transform(max)


df = df.iloc[-24*amount_of_days:]
old_cost = sum(df['Thermal [kW]'].iloc[1:]*price_thermal)+ sum(df['Solar [kW]'].iloc[1:]*price_solar)+sum(df['Wind [kW]'].iloc[1:]*price_wind)+sum(df['Cogeneration [kW]'].iloc[1:]*price_cogen)+sum(df['Hydro [kW]'].iloc[1:]*price_hydro)
old_co2 = sum(df['Total [kW]'].iloc[1:]*df['factorEmisionCO2e'].iloc[1:])
time_list = list(range(1, len(df['Total [kW]'].tolist())))
df['Solar DL [kW]'] = df['Solar [kW]'] / df['Solar [kW] max']
df['Wind DL [kW]'] = df['Wind [kW]'] / df['Wind [kW] max']
df = df.drop(columns=['month', 'year'],axis=1)
df = df.reset_index(drop=True)
# columns =['strategy','price_hydro [$/kWh]','price_solar [$/kWh]','price_wind [$/kWh]','price_thermal [$/kWh]','price_cogen [$/kWh]','co2_hydro [kg/kWh]','co2_solar [kg/kWh]','co2_wind [kg/kWh]','co2_thermal [kg/kWh]','co2_cogen [kg/kWh]','wind_capacity [kW]','solar_capacity [kW]','cost_conventional [$]','co2_conventional [$]','original_cost [$]','original_co2 [kg]']
# df_no_sens = pd.DataFrame(columns=columns)
# df_no_sens.to_csv('normal_data_no_forecast.csv', index=False)


model = ConcreteModel()
model.times = Set(initialize=time_list)

model.solar_capacity = Var(within=NonNegativeReals, bounds = (0,32000000)) #max capacity to determine of sources https://www.sei.org/publications/solar-wind-power-colombia-2022/
model.wind_capacity = Var(within=NonNegativeReals,  bounds=(0,30000000)) #https://www.sei.org/publications/solar-wind-power-colombia-2022/

model.E_co = Var(model.times, within = NonNegativeReals)
model.E_hy = Var(model.times, within = NonNegativeReals)

model.E_th = Var(model.times, within = NonNegativeReals)

model.recalc_wi = Var(model.times, within = NonNegativeReals)
model.recalc_so = Var(model.times, within = NonNegativeReals)
model.old_co2_new_param = Var(model.times, within = NonNegativeReals)
model.old_cost_new_param = Var(model.times, within = NonNegativeReals)
model.new_co2_non_primary = Var(model.times, within = NonNegativeReals) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
model.new_cost_non_primary = Var(model.times, within = NonNegativeReals) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

model.epsilon=Param(initialize=10000,mutable=True) #nodig voor pareto
model.E_tot = Param(model.times, initialize=df['Total [kW]'].iloc[1:].to_dict())
model.E_wi_partial = Param(model.times, initialize=df['Wind DL [kW]'].iloc[1:].to_dict())
model.E_so_partial = Param(model.times, initialize=df['Solar DL [kW]'].iloc[1:].to_dict())
model.E_so_orig = Param(model.times, initialize=df['Solar [kW]'].iloc[1:].to_dict())
model.E_wi_orig = Param(model.times, initialize=df['Wind [kW]'].iloc[1:].to_dict()) 
model.E_th_orig = Param(model.times, initialize=df['Thermal [kW]'].iloc[1:].to_dict())
model.E_co_orig = Param(model.times, initialize=df['Cogeneration [kW]'].iloc[1:].to_dict())
model.E_hy_orig = Param(model.times, initialize=df['Hydro [kW]'].iloc[1:].to_dict())
model.E_co_max = Param(model.times, initialize=df['Cogeneration [kW] max'].iloc[1:].to_dict())
model.E_hy_max = Param(model.times, initialize=df['Hydro [kW] max'].iloc[1:].to_dict())
model.E_th_max = Param(model.times, initialize=df['Thermal [kW] max'].iloc[1:].to_dict())


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

def old_co2_new_param_constr(model, t):
    return model.old_co2_new_param[t] == ((model.E_so_orig[t]*model.CO2_so) +
                                            (model.E_wi_orig[t]*model.CO2_wi) + 
                                            (model.E_th_orig[t]*model.CO2_th) +
                                            (model.E_co_orig[t]*model.CO2_co) +
                                            (model.E_hy_orig[t]*model.CO2_hy))
model.constraint_old_co2_new_param = Constraint(model.times, rule=old_co2_new_param_constr)

def old_cost_new_param_constr(model, t):
    return model.old_cost_new_param[t] == ((model.E_so_orig[t]*model.price_so) +
                                            (model.E_wi_orig[t]*model.price_wi) + 
                                            (model.E_th_orig[t]*model.price_th) +
                                            (model.E_co_orig[t]*model.price_co) +
                                            (model.E_hy_orig[t]*model.price_hy))
model.constraint_old_cost_new_param = Constraint(model.times, rule=old_cost_new_param_constr)

def new_co2_non_primary(model,t): #NOG TOEVOEGEN AAN THESIS
    return model.new_co2_non_primary[t] ==((model.recalc_so[t]*model.CO2_so) +
                                        (model.recalc_wi[t]*model.CO2_wi) +
                                        (model.E_th[t]*model.CO2_th) +
                                        (model.E_co[t]*model.CO2_co) +
                                        (model.E_hy[t]*model.CO2_hy)
    )
model.constraint_new_co2_non_primary = Constraint(model.times, rule=new_co2_non_primary)

def new_cost_non_primary(model,t): #NOG TOEVOEGEN AAN THESIS
    return model.new_cost_non_primary[t] ==((model.recalc_so[t]*model.price_so) +
                                        (model.recalc_wi[t]*model.price_wi) +
                                        (model.E_th[t]*model.price_th) +
                                        (model.E_co[t]*model.price_co) +
                                        (model.E_hy[t]*model.price_hy)
    )
model.constraint_new_cost_non_primary = Constraint(model.times, rule=new_cost_non_primary)

model.objective_cost = Objective(expr=sum((model.E_th[t]*model.price_th) + 
                                    (model.E_co[t]*model.price_co) + 
                                    (model.E_hy[t]*model.price_hy) +
                                    (model.recalc_wi[t]*model.price_wi) +
                                    (model.recalc_so[t]*model.price_so) for t in model.times), sense=minimize)

model.objective_co2_conventional = Objective(expr=sum(((model.recalc_so[t]*model.CO2_so) +
                                            (model.recalc_wi[t]*model.CO2_wi) +
                                            (model.E_th[t]*model.CO2_th) +
                                            (model.E_co[t]*model.CO2_co) +
                                            (model.E_hy[t]*model.CO2_hy)) for t in model.times), sense=minimize)

#COST CONVENTIONAL
model.objective_co2_conventional.deactivate()
results = SolverFactory('glpk').solve(model)
old_costs_new_param = []
old_co2_new_param = []
new_co2 = []

for t in model.times: #KAN ALLEEN ZO DATA UIT HET MODEL HALEN
    old_costs_new_param.append(model.old_cost_new_param[t].value)
    old_co2_new_param.append(model.old_co2_new_param[t].value)
    new_co2.append(model.new_co2_non_primary[t].value)
# ['strategy','price_hydro [$/kWh]','price_solar [$/kWh],'price_wind [$/kWh]','price_thermal [$/kWh]','price_cogen [$/kWh]','co2_hydro [kg/kWh]','co2_solar [kg/kWh]','co2_wind [kg/kWh]','co2_thermal [kg/kWh]','co2_cogen [kg/kWh]','wind_capacity [kW]','solar_capacity [kW]','cost_conventional [$]','co2_conventional [$]','original_cost [$]','original_co2 [kg]']
data = { 
        'strategy': 'cost_conventional',
        'price_hydro [$/kWh]': model.price_hy.value,
        'price_solar [$/kWh]': model.price_so.value,
        'price_wind [$/kWh]': model.price_wi.value,
        'price_thermal [$/kWh]': model.price_th.value,
        'price_cogen [$/kWh]': model.price_co.value,
        'co2_hydro [kg/kWh]': model.CO2_hy.value,
        'co2_solar [kg/kWh]': model.CO2_so.value,
        'co2_wind [kg/kWh]': model.CO2_wi.value,
        'co2_thermal [kg/kWh]': model.CO2_th.value,
        'co2_cogen [kg/kWh]': model.CO2_co.value,
        'wind_capacity [kW]': model.wind_capacity.value,
        'solar_capacity [kW]': model.solar_capacity.value,
        'cost_conventional [$]': model.objective_cost(),
        'co2_conventional [kg]': sum(new_co2),
        'original_cost_new_param [$]': sum(old_costs_new_param),
        'original_co2_new_param [kg]': sum(old_co2_new_param)
        }
print(data)
#make a df out of data
dataframe_cost_conv = pd.DataFrame([data], columns=data.keys())
dataframe_cost_conv.to_csv('cost_conventional.csv')

# new_row = pd.DataFrame(data, index=[0])
# df_no_sens = pd.concat([df_no_sens, new_row], ignore_index=True)
# df_no_sens.tail(1).to_csv('normal_data_no_forecast.csv', mode='a', header=False, index=False)

#CO2 CONVENTIONAL
model.objective_cost.deactivate()
model.objective_co2_conventional.activate()
results = SolverFactory('glpk').solve(model)
new_cost = [] #om de cost te berekenen als side effect na de optimizatie van co2
for t in model.times:
    new_cost.append(model.new_cost_non_primary[t].value)

data = {
        'strategy': 'co2_conventional',
        'price_hydro [$/kWh]': model.price_hy.value,
        'price_solar [$/kWh]': model.price_so.value,
        'price_wind [$/kWh]': model.price_wi.value,
        'price_thermal [$/kWh]': model.price_th.value,
        'price_cogen [$/kWh]': model.price_co.value,
        'co2_hydro [kg/kWh]': model.CO2_hy.value,
        'co2_solar [kg/kWh]': model.CO2_so.value,
        'co2_wind [kg/kWh]': model.CO2_wi.value,
        'co2_thermal [kg/kWh]': model.CO2_th.value,
        'co2_cogen [kg/kWh]': model.CO2_co.value,
        'wind_capacity [kW]': model.wind_capacity.value,
        'solar_capacity [kW]': model.solar_capacity.value,
        'cost_conventional [$]': sum(new_cost),
        'co2_conventional [kg]': model.objective_co2_conventional(),
        'original_cost_new_param [$]': sum(old_costs_new_param),
        'original_co2_new_param [kg]': sum(old_co2_new_param)}
print(data)
df_co2_conv = pd.DataFrame([data], columns=data.keys())
df_co2_conv.to_csv('co2_conventional.csv')
# new_row = pd.DataFrame(data, index=[0])
# df_no_sens = pd.concat([df_no_sens, new_row], ignore_index=True)
# df_no_sens.tail(1).to_csv('normal_data_no_forecast.csv', mode='a', header=False, index=False)
time.sleep(1)
# print(df_no_sens.shape)
