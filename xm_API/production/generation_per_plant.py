from pydataxm import *                          
import datetime as dt
import pandas as pd
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
# import concurrent.futures

import warnings
warnings.filterwarnings('ignore')
objetoAPI = pydataxm.ReadDB()
df = objetoAPI.get_collections()
df = df[df['Entity'] == 'Recurso']

df_generation_plants = pd.read_excel('xm_API\production\Listado_Recursos_Generacion.xlsx',skiprows=3)
df_hydro_plants = df_generation_plants[df_generation_plants['Tipo Generación']=='HIDRAULICA']
df_solar_plants = df_generation_plants[df_generation_plants['Tipo Generación']=='SOLAR']
df_wind_plants = df_generation_plants[df_generation_plants['Tipo Generación']=='EOLICA']
df_thermal_plants = df_generation_plants[df_generation_plants['Tipo Generación']=='TERMICA']
df_cogen_plants = df_generation_plants[df_generation_plants['Tipo Generación']=='COGENERADOR']


plants_hydro = {}
plants_solar = {}
plants_wind = {}
plants_thermal = {}
plants_cogen = {}
plants = {}
list_unknown_assets = []
for i, row in df_generation_plants.iterrows():
    plants[row['Código SIC']] = row['Nombre Recurso']

for i, row in df_hydro_plants.iterrows():
    plants_hydro[row['Código SIC']] = row['Nombre Recurso']

for i, row in df_solar_plants.iterrows():
    plants_solar[row['Código SIC']] = row['Nombre Recurso']

for i, row in df_wind_plants.iterrows():
    plants_wind[row['Código SIC']] = row['Nombre Recurso']

for i, row in df_thermal_plants.iterrows():
    plants_thermal[row['Código SIC']] = row['Nombre Recurso']

for i, row in df_cogen_plants.iterrows():
    plants_cogen[row['Código SIC']] = row['Nombre Recurso']

columnnames_hydro = list(plants_hydro.keys()).append('Timestamp')
columnnames_solar = list(plants_solar.keys()).append('Timestamp')
columnnames_wind = list(plants_wind.keys()).append('Timestamp')
columnnames_thermal = list(plants_thermal.keys()).append('Timestamp')
columnnames_cogen = list(plants_cogen.keys()).append('Timestamp')

def data_to_file(index, definitive_production_df_list):
    #set indexes
    # definitive_solar.set_index('Timestamp', inplace=True)
    # definitive_wind.set_index('Timestamp', inplace=True)
    # definitive_hydro.set_index('Timestamp', inplace=True)
    # definitive_thermal.set_index('Timestamp', inplace=True)
    # definitive_cogen.set_index('Timestamp', inplace=True)
    #set nan to 0
    definitive_solar.fillna(0, inplace=True)
    definitive_wind.fillna(0, inplace=True)
    definitive_hydro.fillna(0, inplace=True)
    definitive_thermal.fillna(0, inplace=True)
    definitive_cogen.fillna(0, inplace=True)

    #add a total production column
    definitive_solar['Total[kW]'] = definitive_solar.sum(axis=1)
    definitive_wind['Total[kW]'] = definitive_wind.sum(axis=1)
    definitive_hydro['Total[kW]'] = definitive_hydro.sum(axis=1)
    definitive_thermal['Total[kW]'] = definitive_thermal.sum(axis=1)
    definitive_cogen['Total[kW]'] = definitive_cogen.sum(axis=1)

    for i in range(len(definitive_production_df_list)):
        # print(i)
        if i == 0:
            type = 'solar'
        elif i == 1:
            type = 'wind'
        elif i == 2:
            type = 'hydro'
        elif i == 3:
            type = 'thermal'
        elif i == 4:
            type = 'cogen'
        definitive_production_df_list[i].to_csv("xm_API\\production"+"\\"+str(type)+"\\"+"month_"+str(index)+".csv")

def process_data(index):
    global start_time, solar, wind, hydro, thermal, cogen, definitive_thermal, definitive_solar, definitive_wind, definitive_hydro, definitive_cogen, objetoAPI, list_unknown_assets
    # global thermal, definitive_thermal, objetoAPI, start_time
    difference = dt.datetime.now() - start_time
    if difference.seconds > 600:
        print("reset connection")
        objetoAPI = pydataxm.ReadDB()
        start_time = dt.datetime.now()  # Reset timer
    else:
        pass
    
    # print("Done months:", index, " of ", len(start_list))
    start_date = start_list[index]
    end_date = end_list[index]

    df_plant = objetoAPI.request_data("Gene", "Recurso", start_date.date(), end_date.date())
    for i in df_plant['Values_code'].unique():
        if i not in list(plants.keys()) and i not in list_unknown_assets:
            list_unknown_assets.append(i)
            print("Unknown asset: ", i, " added to list")
    solar = solar.append(df_plant[df_plant['Values_code'].isin(list(plants_solar.keys()))])
    solar.fillna(0, inplace=True)
    wind = wind.append(df_plant[df_plant['Values_code'].isin(list(plants_wind.keys()))])
    wind.fillna(0, inplace=True)
    hydro = hydro.append(df_plant[df_plant['Values_code'].isin(list(plants_hydro.keys()))])
    hydro.fillna(0, inplace=True)

    thermal = thermal.append(df_plant[df_plant['Values_code'].isin(list(plants_thermal.keys()))]) #OF HIER EEN FOUT
    thermal.fillna(0, inplace=True)
    cogen = cogen.append(df_plant[df_plant['Values_code'].isin(list(plants_cogen.keys()))])
    cogen.fillna(0, inplace=True)

    for i in thermal['Values_code'].unique():
        temp_thermal = thermal[thermal['Values_code'] == i].copy()
        temp_thermal = pd.melt(temp_thermal, id_vars=['Date'], value_vars=[f'Values_Hour{i:02d}' for i in range(1, 24)], var_name='Hour', value_name=i)
        temp_thermal['Timestamp'] = pd.to_datetime(temp_thermal['Date']) + pd.to_timedelta(temp_thermal['Hour'].str.split('Hour').str[-1].astype(int), unit='h')
        temp_thermal.drop(columns=['Date', 'Hour'], inplace=True)
        temp_thermal.rename(columns={'Values_code': i}, inplace=True)
        for j in definitive_thermal.columns:
            if j not in temp_thermal.columns:
                temp_thermal[j] = 0
        temp_thermal.set_index('Timestamp', inplace=True)
        
        if definitive_thermal.empty:
            definitive_thermal = temp_thermal
        else:
            definitive_thermal = pd.concat([definitive_thermal, temp_thermal], axis=0)
            definitive_thermal.sort_values(by=['Timestamp'], inplace=True)

    for i in solar['Values_code'].unique():
        temp_solar = solar[solar['Values_code'] == i].copy()
        temp_solar = pd.melt(temp_solar, id_vars=['Date'], value_vars=[f'Values_Hour{i:02d}' for i in range(1, 24)], var_name='Hour', value_name=i)
        temp_solar['Timestamp'] = pd.to_datetime(temp_solar['Date']) + pd.to_timedelta(temp_solar['Hour'].str.split('Hour').str[-1].astype(int), unit='h')
        temp_solar.drop(columns=['Date', 'Hour'], inplace=True)
        temp_solar.rename(columns={'Values_code': i}, inplace=True)
        for j in definitive_solar.columns:
            if j not in temp_solar.columns:
                temp_solar[j] = 0
        temp_solar.set_index('Timestamp', inplace=True)
        
        if definitive_solar.empty:
            definitive_solar = temp_solar
        else:
            definitive_solar = pd.concat([definitive_solar, temp_solar], axis=0)
            definitive_solar.sort_values(by=['Timestamp'], inplace=True)

    for i in wind['Values_code'].unique():
        temp_wind = wind[wind['Values_code'] == i].copy()
        temp_wind = pd.melt(temp_wind, id_vars=['Date'], value_vars=[f'Values_Hour{i:02d}' for i in range(1, 24)], var_name='Hour', value_name=i)
        temp_wind['Timestamp'] = pd.to_datetime(temp_wind['Date']) + pd.to_timedelta(temp_wind['Hour'].str.split('Hour').str[-1].astype(int), unit='h')
        temp_wind.drop(columns=['Date', 'Hour'], inplace=True)
        temp_wind.rename(columns={'Values_code': i}, inplace=True)
        for j in definitive_wind.columns:
            if j not in temp_wind.columns:
                temp_wind[j] = 0
        temp_wind.set_index('Timestamp', inplace=True)

        if definitive_wind.empty:
            definitive_wind = temp_wind
        else:
            definitive_wind = pd.concat([definitive_wind, temp_wind], axis=0)
            definitive_wind.sort_values(by=['Timestamp'], inplace=True)
    
    for i in hydro['Values_code'].unique():
        temp_hydro = hydro[hydro['Values_code'] == i].copy()
        temp_hydro = pd.melt(temp_hydro, id_vars=['Date'], value_vars=[f'Values_Hour{i:02d}' for i in range(1, 24)], var_name='Hour', value_name=i)
        temp_hydro['Timestamp'] = pd.to_datetime(temp_hydro['Date']) + pd.to_timedelta(temp_hydro['Hour'].str.split('Hour').str[-1].astype(int), unit='h')
        temp_hydro.drop(columns=['Date', 'Hour'], inplace=True)
        temp_hydro.rename(columns={'Values_code': i}, inplace=True)
        for j in definitive_hydro.columns:
            if j not in temp_hydro.columns:
                temp_hydro[j] = 0
        temp_hydro.set_index('Timestamp', inplace=True)
        if definitive_hydro.empty:
            definitive_hydro = temp_hydro
        else:
            definitive_hydro = pd.concat([definitive_hydro, temp_hydro], axis=0)
            definitive_hydro.sort_values(by=['Timestamp'], inplace=True)
    
    for i in cogen['Values_code'].unique():
        temp_cogen = cogen[cogen['Values_code'] == i].copy()
        temp_cogen = pd.melt(temp_cogen, id_vars=['Date'], value_vars=[f'Values_Hour{i:02d}' for i in range(1, 24)], var_name='Hour', value_name=i)
        temp_cogen['Timestamp'] = pd.to_datetime(temp_cogen['Date']) + pd.to_timedelta(temp_cogen['Hour'].str.split('Hour').str[-1].astype(int), unit='h')
        temp_cogen.drop(columns=['Date', 'Hour'], inplace=True)
        temp_cogen.rename(columns={'Values_code': i}, inplace=True)
        for j in definitive_cogen.columns:
            if j not in temp_cogen.columns:
                temp_cogen[j] = 0
        temp_cogen.set_index('Timestamp', inplace=True)

        if definitive_cogen.empty:
            definitive_cogen = temp_cogen
        else:
            definitive_cogen = pd.concat([definitive_cogen, temp_cogen], axis=0)
            definitive_cogen.sort_values(by=['Timestamp'], inplace=True)


start_time = dt.datetime.now()
solar = pd.DataFrame()
wind = pd.DataFrame()
hydro = pd.DataFrame()
thermal = pd.DataFrame()
cogen = pd.DataFrame()
change_variable = False
definitive_thermal = pd.DataFrame(columns = columnnames_thermal)
definitive_solar = pd.DataFrame(columns = columnnames_solar)
definitive_wind = pd.DataFrame(columns = columnnames_wind)
definitive_hydro = pd.DataFrame(columns = columnnames_hydro)
definitive_cogen = pd.DataFrame(columns = columnnames_cogen)

definitive_production_df_list = [definitive_solar, definitive_wind, definitive_hydro, definitive_thermal, definitive_cogen]
months = 67
start_year = 2018
start_date = dt.datetime(start_year, 1, 1)
start_list = [start_date + relativedelta(months=i) for i in range(0, months)] 
end_list = [start_date + relativedelta(months=i+1, days=-1) for i in range(0, months)]
objetoAPI = pydataxm.ReadDB()
max_threads = 8 
# with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
for index, start in enumerate(start_list):
    if index <64: #25 mogelijks slecht? Ook checken of alles voor 25 wel goed gelukt was... 
        print("skip month:", index, " of ", len(start_list))
    else:
        now = dt.datetime.now()
        difference = now - start_time
        print("Seconds elapsed: ",difference.seconds)
        if difference.seconds > 600:
            print("reset connection")
            objetoAPI = pydataxm.ReadDB()
            start_time = dt.datetime.now()
        # executor.map(process_data(index), range(len(start_list)))
        process_data(index)
        
        # if index in [5,10,15,20,25,30,35,40,45,50,55,60,67]: #control if deleting frees up ram
        # if index in [2,5,10]:
        definitive_production_df_list = [definitive_solar, definitive_wind, definitive_hydro, definitive_thermal, definitive_cogen]
        print("Wait for stop untill 'done months' is printed")
        data_to_file(index, definitive_production_df_list)
        print("Done months:", index, " of ", len(start_list))
        definitive_thermal.drop(definitive_thermal.index, inplace=True)
        definitive_thermal = pd.DataFrame(columns = columnnames_thermal)
        definitive_solar.drop(definitive_solar.index, inplace=True)
        definitive_solar = pd.DataFrame(columns = columnnames_solar)
        definitive_wind.drop(definitive_wind.index, inplace=True)
        definitive_wind = pd.DataFrame(columns = columnnames_wind)
        definitive_hydro.drop(definitive_hydro.index, inplace=True)
        definitive_hydro = pd.DataFrame(columns = columnnames_hydro)
        definitive_cogen.drop(definitive_cogen.index, inplace=True)
        definitive_cogen = pd.DataFrame(columns = columnnames_cogen)
        print("---- Files", index, " written and RAM freed ----")

print("Unknown assets: ", list_unknown_assets)