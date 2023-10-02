import pandas as pd
import datetime
from pandas import Timestamp
import numpy as np
from pyomo.environ import *
df = pd.read_csv("C:\\Users\\LIAT-ER\\Documents\\Thesis_Thibo\\production_2\\production_only_totals.csv")
price_hydro = 60 #USD/MWh
price_solar = 78 #USD/MWh
price_wind = 59 #USD/MWh
price_thermal = 150 #USD/MWh  NOG AANPASSEN
price_cogen = 100 #USD/MWh NOG AANPASSEN
co2_hydro = 24 #kg/MWh
co2_solar = 45 #kg/MWh
co2_wind = 11 #kg/MWh
co2_thermal = 700 #kg/MWh
co2_cogen = 600/2 #kg/MWh 

original_total_co2 = (df['Hydro [kW]'].sum()/1000)*co2_hydro + (df['Solar [kW]'].sum()/1000)*co2_solar + (df['Wind [kW]'].sum()/1000)*co2_wind + (df['Thermal [kW]'].sum()/1000)*co2_thermal + (df['Cogeneration [kW]'].sum()/1000)*co2_cogen
original_total_price = (df['Hydro [kW]'].sum()/1000)*price_hydro + (df['Solar [kW]'].sum()/1000)*price_solar + (df['Wind [kW]'].sum()/1000)*price_wind + (df['Thermal [kW]'].sum()/1000)*price_thermal + (df['Cogeneration [kW]'].sum()/1000)*price_cogen
print('Total kilograms of CO2:',original_total_co2)
print('Total price in USD:',original_total_price)

model = ConcreteModel()

model.solar_increase = Var(within=NonNegativeReals, bounds=(0, 0.20))
model.wind_increase = Var(within=NonNegativeReals, bounds=(0, 0.5))


def cost_expr_rule(model):
    total_cost = 0
    for i in df.index:
        thermal_contribution = df.loc[i, 'Thermal [kW]'] - ((df.loc[i, 'Solar [kW]'] * (1 + model.solar_increase)) + (df.loc[i, 'Wind [kW]'] * (1 + model.wind_increase)))
        if thermal_contribution < 0:
            # If thermal would go below 0, decrease hydro by the amount exceeding 0
            # hydro_contribution = max(0, -thermal_contribution)
            hydro_contribution = -thermal_contribution if -thermal_contribution > 0 else 0
            thermal_contribution = 0
        else:
            hydro_contribution = 0
        total_cost += (
            (price_solar * df.loc[i, 'Solar [kW]'] * (1 + model.solar_increase)) + 
           (price_wind * df.loc[i, 'Wind [kW]'] * (1 + model.wind_increase)) + 
            (price_thermal * thermal_contribution) + 
            (price_hydro * (df.loc[i, 'Hydro [kW]'] - hydro_contribution)) +
            (price_cogen * df.loc[i, 'Cogeneration [kW]'])
        )
    return total_cost

model.cost_expr2 = Expression(rule=cost_expr_rule)

model.objective = Objective(expr=model.cost_expr2, sense=minimize)

results = SolverFactory('glpk').solve(model)

print('Solar increase:',model.solar_increase())
print('Wind increase:',model.wind_increase())

print('Difference in price:',original_total_price - model.objective())
print('Procentual :',((original_total_price - model.objective())/original_total_price)*100)
