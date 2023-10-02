import pandas as pd
import datetime
import numpy as np
from pyomo.environ import *
df = pd.read_csv("C:\\Users\\thibo\\OneDrive - Hanzehogeschool Groningen\\Documenten\\Github\\thesis_UNAL\\xm_API\\production\\production_only_totals.csv")
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

df = df.iloc[-24*30:]

time_list = list(range(1, len(df['Total [kW]'].tolist())))
df['Solar DL [kW]'] = df['Solar [kW]'] / 10000 # Hier bvb de maximale capaciteit per maand gaan instellen
df['Wind DL [kW]'] = df['Wind [kW]'] / 30000 #Hier zelfde als hierboven
#kolom toevoegen die per maand de maximale power meegeeft(=maximale capaciteit) en hierboven ingeven.


df = df.reset_index(drop=True)

print(df['Wind DL [kW]'])
model = ConcreteModel()
model.times = Set(initialize=time_list)

model.solar_increase = Var(within=NonNegativeReals)
model.wind_increase = Var(within=NonNegativeReals,  bounds=(0,50000000))
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
model.price_th = Param(initialize=price_thermal)
model.price_co = Param(initialize=price_cogen)
model.price_hy = Param(initialize=price_hydro)
model.price_so = Param(initialize=price_solar)
model.price_wi = Param(initialize=price_wind)

def recalc_wi_rule(model, t):
    return model.recalc_wi[t] == model.production_wi_profile[t]*model.wind_increase
model.recalc_wi_constraint = Constraint(model.times, rule=recalc_wi_rule)

def recalc_so_rule(model, t):
    return model.recalc_so[t] == model.production_so_profile[t]*model.solar_increase
model.recalc_so_constraint = Constraint(model.times, rule=recalc_so_rule)


def demand_rule(model, t):
    return model.production_th[t] + model.production_co[t] + model.production_hy[t] + model.recalc_so[t] + model.recalc_wi[t] == model.production_total[t]
model.demand_constraint = Constraint(model.times, rule=demand_rule)



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
print(model.solar_increase.value)
print(model.wind_increase.value)