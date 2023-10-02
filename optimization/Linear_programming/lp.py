import pandas as pd
import datetime
import numpy as np
from pyomo.environ import *
# df = pd.read_csv("C:\\Users\\LIAT-ER\\Documents\\Thesis_Thibo\\production_2\\production_only_totals.csv")
df = pd.read_csv("C:\\Users\\thibo\\OneDrive - Hanzehogeschool Groningen\\Documenten\\Github\\thesis_UNAL\\xm_API\\production\\production_only_totals.csv")
# df = pd.read_excel("C:\\Users\\thibo\\OneDrive - Hanzehogeschool Groningen\\Documenten\\Github\\thesis_UNAL\\optimization\\Linear_programming\\easy_dataset.xlsx",)

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
print(df.head())

# time_list = df['Timestamp'].tolist() => met timestamp object wil ie nie werken
time_list = []
for i in range(len(time_list)):
        time_list[i] = i
production_list_co = df['Cogeneration [kW]'].tolist()
production_list_hy = df['Hydro [kW]'].tolist()
production_list_so = df['Solar [kW]'].tolist()
production_list_wi = df['Wind [kW]'].tolist()
production_list_th = df['Thermal [kW]'].tolist()
production_list_total = df['Total [kW]'].tolist()

model = ConcreteModel()
model.solar_increase = Var(within=NonNegativeReals, bounds=(0, 1000), initialize=0) #normaal op 0-100 maar mag nu overdreven groot zijn zodat de rules zeker werken
model.wind_increase = Var(within=NonNegativeReals, bounds=(0, 500), initialize=0) #normaal op 0-50 maar same als hierboven
model.times = Set(initialize=time_list)
model.production_co = Set(initialize=production_list_co)
model.production_hy = Set(initialize=production_list_hy)
model.production_so = Set(initialize=production_list_so)
model.production_wi = Set(initialize=production_list_wi)
model.production_th = Set(initialize=production_list_th)
model.production_total = Set(initialize=production_list_total)
#alle sets lijken mij goed te zijn

def eth_rule(model, t): #kijken dat thermische productie niet negatief is, indien wel ==>0 en rest weglaten
        if model.production_th[t] - ((model.solar_increase/100) * model.production_so[t]) - ((model.wind_increase/100) * model.production_wi[t]) <0:
                return model.production_th[t] == 0
        else:
                return model.production_th[t] - ((model.solar_increase/100) * model.production_so[t]) - ((model.wind_increase/100) * model.production_wi[t])
model.eth_constraint = Constraint(model.times, rule=eth_rule)

def total_rule(model,t): #kijken of nog steeds voldaan wordt aan de zelfde totale productie voor ieder moment
        #moet bij total_new ook een (t) komen aangezien het eigenlijk alleen een variabele is voor een bepaald moment
        #moet enkel een constraint zijn voor de momenten waarop de totale productie negatief is
        total_new = model.production_total[t] - ((model.solar_increase/100) * model.production_so[t]) - ((model.wind_increase/100) * model.production_wi[t]) - (model.production_co[t]) - (model.production_hy[t]) -(model.production_th[t])
        if total_new < 0:
            return total_new == 0
        else:
               Constraint.Skip
model.total_constraint = Constraint(model.times, rule=total_rule)

#onderstaande moet wel kloppen
model.objective = Objective(expr=sum(
        (model.production_co[t]*price_cogen) + 
        (model.production_hy[t]*price_hydro) + 
        (model.production_so[t]*price_solar) +
        (model.production_wi[t]*price_wind) +
        (model.production_th[t]*price_thermal)
        for t in model.times), sense=minimize)

solver = SolverFactory('glpk')
results = solver.solve(model)

print(results)