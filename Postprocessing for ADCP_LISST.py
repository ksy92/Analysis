# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 20:44:19 2020

@author: Siyoon Kwon
"""
import pandas as pd
import spectral
from spectral import*
import spectral.io.envi as envi 
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import matplotlib.font_manager as font_manager
from tqdm import tqdm
import re
from shapely.geometry import Point
from geopandas import GeoDataFrame
from haversine import haversine
import pickle 

#%% Reading dataset

file_path = os.getcwd()
os.chdir(file_path)

path = 'D:\부유사 현장 실험\\210330 황강합류부\LISST_analysis'
ADCP = pd.read_csv(path+'\ADCP_HSI_Area.csv')
LISST = pd.read_csv(path+'\LISST for matching.csv')

#%% Segmentation of time in ADCP data

adcp_t = ADCP['DateTime']
a = re.findall("\d+", adcp_t.iloc[1])
ADCP['Hour']= ADCP['DateTime']
ADCP['Minute']= ADCP['DateTime']
ADCP['Second']= ADCP['DateTime']

for i in tqdm(np.arange(len(ADCP))):
              ADCP['Hour'][i]=re.findall("\d+", adcp_t.iloc[i])[3]
              ADCP['Minute'][i]=re.findall("\d+", adcp_t.iloc[i])[4]
              ADCP['Second'][i]=re.findall("\d+", adcp_t.iloc[i])[5]

Merged_dat = LISST.copy()

#%% Time matching of two sensors            
                    
ADCP_copy = ADCP.copy()
ADCP_copy['Hour'] = ADCP_copy['Hour'].astype(str) 
ADCP_copy['Minute'] = ADCP_copy['Minute'].astype(str) 
ADCP_copy['Second'] = ADCP_copy['Second'].astype(str) 

LISST_copy = LISST.copy()
LISST_copy['Hour_'] = LISST_copy['Hour'].astype(str) 
LISST_copy['Minute_'] = LISST_copy['Minute'].astype(str) 
LISST_copy['Second_'] = LISST_copy['Second'].astype(str) 

LISST_ADCP = LISST_copy.copy()
LISST_ADCP[ADCP_copy.columns.values] = pd.DataFrame(np.zeros([len(LISST_copy), len(ADCP_copy.columns.values)]))

for i in tqdm(range(len(LISST_copy))):
    cur_hour = LISST_copy.iloc[i]['Hour_']
    cur_min = LISST_copy.iloc[i]['Minute_']
    cur_sec = LISST_copy.iloc[i]['Second_']
    
    adcp_row = ADCP_copy.loc[(ADCP_copy['Hour'] == cur_hour) & (ADCP_copy['Minute'] == cur_min) & (ADCP_copy['Second'] == cur_sec)]
    # adcp_idx = ADCP_copy.loc[ADCP_copy['ttime'] == cur_time].index[0]
    
    if len(adcp_row) != 0 : 
        LISST_ADCP.iloc[i, len(LISST_copy.columns.values):] = adcp_row

#%%Delte raws containing zero

SUM = LISST_ADCP[ADCP_copy.columns.values]
SUM = SUM.astype({'Unit': str})

# LISST_ADCP_copy = LISST_ADCP.copy()
muyaho = []
for i in np.arange(len(SUM)):
    cur_meter_str = SUM.iloc[i]['Unit']
    if len(cur_meter_str) < 5 : 
        muyaho.append(i)        

LISST_ADCP = LISST_ADCP.drop(muyaho)

#%%Save to csv file 

LISST_ADCP.to_csv('LISST_ADCP.csv')


#%% Groupby index for Spatial averaging 

crd_ls = []
for i in range(len(LISST_ADCP)):
    x_crd = LISST_ADCP.iloc[i]['Latitude']
    y_crd = LISST_ADCP.iloc[i]['Longitude']
    crd_pt = Point(y_crd, x_crd)
    crd_ls.append(crd_pt)

LISST_ADCP = LISST_ADCP[['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36', 
                            'D50','SSC', 'WaterTemp', 'Latitude', 'Longitude', 'MeanDepth','VertiDepth']]





LISST_ADCP['geometry'] = crd_ls
LISST_ADCP_gdf = GeoDataFrame(LISST_ADCP, crs="EPSG:4326", geometry='geometry')
LISST_ADCP_gdf1 = LISST_ADCP_gdf.copy()
LISST_ADCP_gdf2 = LISST_ADCP_gdf.copy()



avg_dict = dict()

for i in tqdm(range(len(LISST_ADCP_gdf1))): 
    curi_x = LISST_ADCP_gdf1.iloc[i].geometry.x
    curi_y = LISST_ADCP_gdf1.iloc[i].geometry.y
    
    avg_ls = [] 
    
    if i > 0: 
        temp = np.concatenate(list(avg_dict.values()))

        if i not in temp: 
            for j in range(i, len(LISST_ADCP_gdf2)):
                curj_x = LISST_ADCP_gdf2.iloc[j].geometry.x
                curj_y = LISST_ADCP_gdf2.iloc[j].geometry.y
        
                dist = haversine((curi_y, curi_x), (curj_y, curj_x), unit = 'm')
                
                if (dist <= 5) & (dist != 0) : 
                    avg_ls.append(j)
            
            avg_dict[i] = avg_ls
    
    else: 
       
        for j in range(i, len(LISST_ADCP_gdf2)):
            curj_x = LISST_ADCP_gdf2.iloc[j].geometry.x
            curj_y = LISST_ADCP_gdf2.iloc[j].geometry.y
    
            dist = haversine((curi_y, curi_x), (curj_y, curj_x), unit = 'm')
            
            if (dist <= 3) & (dist != 0) : 
                avg_ls.append(j)
        
        avg_dict[i] = avg_ls
        
def write_data(data, name):
    with open(name + '.bin', 'wb') as f:
        pickle.dump(data, f)
        
def load_data(name):
    with open(name + '.bin', 'rb') as f:
        data = pickle.load(f)
    return data  

# write_data(avg_dict, 'avg_dict_3')
# write_data(avg_dict, 'avg_dict_5')
avg_dict_5 = load_data('avg_dict_5')
avg_dict_3 = load_data('avg_dict_3')

# if __name__=='__main__':
#     pool = Pool(8)
#     avg_dict = pool.map(find_near_idx, (LISST_ADCP_gdf1, LISST_ADCP_gdf2, 5))
#     # time.sleep(1)
    
#%% Spatial averaging 
from shapely import geometry

#list(zip(avg_dict_5.values(), avg_dict_5.keys()))
avg_dat = pd.DataFrame(np.zeros([len(avg_dict_5),len(LISST_ADCP_gdf.iloc[0,:])]), columns =LISST_ADCP_gdf.columns.values, dtype=float)

avg_dat['geometry']=Point(0,0)
avg_dat_gdf = GeoDataFrame(avg_dat, crs="EPSG:4326", geometry='geometry')


for zz in tqdm(np.arange(len(avg_dict_5))):
    zzzz = list(zip(avg_dict_5.values(), avg_dict_5.keys()))
    idx = np.concatenate((np.array([zzzz[zz][1]]),np.array(zzzz[zz][0])),axis=0)
    avg_dat_gdf.iloc[zz,:-1] = LISST_ADCP_gdf.iloc[list(idx),:].mean(axis=0)
    if len(idx) >2:
        poly = geometry.Polygon([[p.x, p.y] for p in LISST_ADCP_gdf.iloc[list(idx),-1]])
        avg_dat_gdf.iloc[zz,-1] = poly.centroid
    elif len(idx) == 2:
        line = geometry.LineString([[p.x, p.y] for p in LISST_ADCP_gdf.iloc[list(idx),-1]])
        avg_dat_gdf.iloc[zz,-1] = line.centroid
        
    else:
        avg_dat_gdf.iloc[zz,-1] = LISST_ADCP_gdf.iloc[list(idx),-1]
        
        
# print(poly.wkt)

#%%Save to csv file 

avg_dat_gdf.to_csv('LISST_ADCP_spatial_avg.csv')
