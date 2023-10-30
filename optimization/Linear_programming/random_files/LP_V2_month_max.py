import pandas as pd
import datetime
import numpy as np
from pyomo.environ import *
import matplotlib.pyplot as plt
df = pd.read_csv("C:\\Users\\thibo\\OneDrive - Hanzehogeschool Groningen\\Documenten\\Github\\UNAL_thesis\\xm_API\\production\\production_only_totals.csv")
df = df.fillna(0) #controleren!
price_hydro = 65 #USD/MWh
price_solar = 60 #USD/MWh
price_wind = 59 #USD/MWh
price_thermal = 150 #USD/MWh  NOG AANPASSEN
price_cogen = 100 #USD/MWh NOG AANPASSEN
co2_hydro = 24 #kg/MWh
co2_solar = 45 #kg/MWh
co2_wind = 11 #kg/MWh
co2_thermal = 700 #kg/MWh
co2_cogen = 600/2 #kg/MWh 
columns_to_determine_monthly_max = ['Cogeneration [kW]', 'Hydro [kW]', 'Solar [kW]', 'Wind [kW]', 'Thermal [kW]']
#add a column which shows the max of that current month and year for each of the columns in columns_to_determine_monthly_max
df['month'] = pd.DatetimeIndex(df['Timestamp']).month
df['year'] = pd.DatetimeIndex(df['Timestamp']).year
for column in columns_to_determine_monthly_max:
    df[column + ' max'] = df.groupby(['month', 'year'])[column].transform(max)



#plot Cogeneration [kW] and Cogeneration [kW] max to see if it works
# plt.plot(df['Cogeneration [kW]'], label='Cogeneration [kW]')
# plt.plot(df['Cogeneration [kW] max'], label='Cogeneration [kW] max')
# plt.legend()
# plt.show()

# plt.plot(df['Hydro [kW]'], label='Hydro [kW]')
# plt.plot(df['Hydro [kW] max'], label='Hydro [kW] max')
# plt.legend()
# plt.show()

# plt.plot(df['Solar [kW]'], label='Solar [kW]')
# plt.plot(df['Solar [kW] max'], label='Solar [kW] max')
# plt.legend()
# plt.show()

# plt.plot(df['Wind [kW]'], label='Wind [kW]')
# plt.plot(df['Wind [kW] max'], label='Wind [kW] max')
# plt.legend()
# plt.show()

# plt.plot(df['Thermal [kW]'], label='Thermal [kW]')
# plt.plot(df['Thermal [kW] max'], label='Thermal [kW] max')
# plt.legend()
# plt.show()

#uitbreidingen
#1) die maximale capaciteit per maand meegeven wind en zon (check zie grafieken boven)
#2) constraints toevoegen waardoor alle andere units niet hoger dan hun max kunnen gaan (check)
#3) voeg toe dat de totale productie van een nieuwe maand niet hoger kan zijn dan de originele totale productie van hydro

df = df.iloc[-24*30:]
time_list = list(range(1, len(df['Total [kW]'].tolist())))
df['Solar DL [kW]'] = df['Solar [kW]'] / df['Solar [kW] max']
df['Wind DL [kW]'] = df['Wind [kW]'] / df['Wind [kW] max']
df = df.drop(columns=['month', 'year'],axis=1)
# print(df.head(10))

df = df.reset_index(drop=True)

model = ConcreteModel()
model.times = Set(initialize=time_list)

model.solar_capacity = Var(within=NonNegativeReals) #max capacity to determine of sources
model.wind_capacity = Var(within=NonNegativeReals,  bounds=(0,50000000))

model.production_co = Var(model.times, within = NonNegativeReals)
model.production_hy = Var(model.times, within = NonNegativeReals)
model.production_so = Var(model.times, within = NonNegativeReals)
model.production_wi = Var(model.times, within = NonNegativeReals)
model.production_th = Var(model.times, within = NonNegativeReals)

model.recalc_wi = Var(model.times, within = NonNegativeReals)
model.recalc_so = Var(model.times, within = NonNegativeReals)


model.production_total = Param(model.times, initialize=df['Total [kW]'].iloc[1:].to_dict())
model.production_wi_profile = Param(model.times, initialize=df['Wind DL [kW]'].iloc[1:].to_dict())
model.production_so_profile = Param(model.times, initialize=df['Solar DL [kW]'].iloc[1:].to_dict())
model.max_production_co = Param(model.times, initialize=df['Cogeneration [kW] max'].iloc[1:].to_dict())
model.max_production_hy = Param(model.times, initialize=df['Hydro [kW] max'].iloc[1:].to_dict())
model.max_production_th = Param(model.times, initialize=df['Thermal [kW] max'].iloc[1:].to_dict())
model.price_th = Param(initialize=price_thermal)
model.price_co = Param(initialize=price_cogen)
model.price_hy = Param(initialize=price_hydro)
model.price_so = Param(initialize=price_solar)
model.price_wi = Param(initialize=price_wind)

def recalc_wi_rule(model, t):
    return model.recalc_wi[t] == model.production_wi_profile[t]*model.wind_capacity

model.recalc_wi_constraint = Constraint(model.times, rule=recalc_wi_rule)

def recalc_so_rule(model, t):
    return model.recalc_so[t] == model.production_so_profile[t]*model.solar_capacity
model.recalc_so_constraint = Constraint(model.times, rule=recalc_so_rule)


def demand_rule(model, t):
    return model.production_th[t] + model.production_co[t] + model.production_hy[t] + model.recalc_so[t] + model.recalc_wi[t] == model.production_total[t]
model.demand_constraint = Constraint(model.times, rule=demand_rule)

def max_constraint_co(model, t):
    return model.production_co[t] <= df['Cogeneration [kW] max'].iloc[t]
model.max_constraint_co = Constraint(model.times, rule=max_constraint_co)

def max_constraint_hy(model, t):
    return model.production_hy[t] <= df['Hydro [kW] max'].iloc[t]
model.max_constraint_hy = Constraint(model.times, rule=max_constraint_hy)

def max_constraint_th(model, t):
    return model.production_th[t] <= df['Thermal [kW] max'].iloc[t]



model.objective = Objective(expr=sum((model.production_th[t]*model.price_th) + 
                                     (model.production_co[t]*model.price_co) + 
                                     (model.production_hy[t]*model.price_hy) +
                                     (model.recalc_wi[t]*model.price_wi) +
                                     (model.recalc_so[t]*model.price_so) for t in model.times), sense=minimize)
solver = SolverFactory('glpk')
results = solver.solve(model)

for t in model.times:
    if t<5:
        print(t)
        print(model.recalc_wi[t].value)
        print(model.production_hy[t].value)
        print(model.recalc_so[t].value)
print('---------')
print('Desired solar capacity: ',model.solar_capacity.value)
print('Desired wind capacity: ',model.wind_capacity.value)