import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from scipy.io import loadmat

root = tk.Tk()
root.withdraw()

matfile_path = filedialog.askopenfilename()
appendixC_path = filedialog.askopenfilename()
#features_path = filedialog.askopenfilename()

#matfile_path = 'D:/Synology Drive Local/Projects/Projects/4. Europe/Estonia/Rosen/22-AN-EST-ROS-ELR-01/4 Data Analysis/D5/Original Project/project2.mat'
#appendixC_path = 'D:/Synology Drive Local/Projects/Projects/4. Europe/Estonia/Rosen/22-AN-EST-ROS-ELR-01/4 Data Analysis/D5/Original Project/Results/appendixC.xlsx'
#features_path = 'D:/Synology Drive Local/Projects/Projects/4. Europe/Estonia/Rosen/22-AN-EST-ROS-ELR-01/4 Data Analysis/D5/Interpretation/TOPO.xlsx'

survey_df = loadmat(matfile_path)
anomaly_df = pd.read_excel(appendixC_path, sheet_name='Anomaly Table')
#features_df = pd.read_excel(features_path)

anomaly_df['Dist'] = ''
anomaly_df['Topo ID'] = ''
anomaly_df['Topo Features'] = ''

#Calculate distance between two points
def haversine_vectorize(lon1, lat1, lon2, lat2):
    """Returns distance, in kilometers, between one set of longitude/latitude coordinates and another"""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
 
    newlon = lon2 - lon1
    newlat = lat2 - lat1
 
    haver_formula = np.sin(newlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(newlon/2.0)**2
 
    dist = 2 * np.arcsin(np.sqrt(haver_formula ))
    km = 6367 * dist #6367 for distance in KM for miles use 3958
    return km * 1000

#Set color for anomaly_df based on %SMYS
def SetColor(y):
        if(y < 40):
            return "black"
        elif(y >= 40 and y <= 50):
            return "blue"
        elif(y >= 50 and y <= 60):
            return "green"
        elif(y >= 60 and y <= 72):
            return "yellow"
        else:
            return "red"
        
#SCZ Deletion
def SCZDeletion():
    dis = anomaly_df.iloc[:, 1].round(decimals=2)
    Str = anomaly_df.iloc[:, 6].round(decimals=0)
    
    sizedis = np.size(dis)
    thresh = 1.2
    k = 1
    j = 0

    outres = np.zeros(sizedis)
    maxes = np.zeros(sizedis)
    outrest = np.zeros(sizedis)

    while j < (sizedis - 3):
        j = j + 1
        while (dis[j] - dis[j-1]) < thresh:
            outres[j-1] = k
            outres[j] = k
            j = j + 1
            if j > sizedis - 1:
                break        
        if j > sizedis - 2:
            j = j - 1
        else:
            if (dis[j+1] - dis[j]) < thresh:
                k = k + 1
            
    for j in range(1, sizedis - 2):
        if outres[j] > 0:
            if outres[j] == outres[j+1]:
                if Str[j] > Str[j+1]:
                    if Str[j-1] > Str[j] and outres[j-1] == outres[j]:
                        maxes[j] = 1
                        maxes[j+1] = 1
                    else:
                        maxes[j] = 2
                        maxes[j+1] = 1
                else:
                    if outres[j+1] == outres[j+2]:
                        maxes[j+1] = 2
                        maxes[j] = 1
                    else:
                        if Str[j] < Str[j+1]:
                            maxes[j] = 1;
                            maxes[j+1] = 2

    for j in range(0, sizedis):
        if maxes[j] == 2:
            outrest[j] = 0
        else:
            if outres[j] > 0:
                outrest[j] = 1
            
    if dis[1] - dis[0] < thresh:
        if Str[0] > Str[1]:
            outrest[0] = 0
        else:
            outrest[0] = 1
            
    if dis[sizedis - 1] - dis[sizedis - 2] < thresh:
        if Str[sizedis - 1] > Str[sizedis - 2]:
            outrest[sizedis - 1] = 0
        else:
            outrest[sizedis - 1] = 1
            
    anomaly_df['Delete'] = outrest
    anomaly_df['Delete'].replace(0, np.nan, inplace=True)

#listmatcher
#def listmatcher():
#    searchfor = ['CP', 'F', 'MH', 'MP', 'MF', 'MO', 'PL', 'TM', 'VF', 'VH' ]
#    search_df = features_df[features_df['Code'].str.contains('|'.join(searchfor))]
#    
#    for i in range(0, anomaly_df.shape[0]):
#        sepdis = []
#        for j in search_df.index:
#            sepdis.append(haversine_vectorize(anomaly_df['Longitude [WGS84]'][i], anomaly_df['Latitude [WGS84]'][i], search_df['WGS84 Longitude'][j], search_df['WGS84 Latitude'][j]))
#        anomaly_df['Dist'][i] = min(sepdis)
#        anomaly_df['Topo ID'][i] = search_df['Name'].iloc[np.argmin(sepdis)]
#        anomaly_df['Topo Features'][i] = search_df['Code'].iloc[np.argmin(sepdis)]
#        
#   anomaly_df.loc[anomaly_df['Dist'] < 2.5, 'Comments'] = "AMO"


def plotStripChart(anomaly_df):
    distance = survey_df['SurveyMainData']['Distance'][0][0].flatten()
    
    fig = make_subplots(rows=8, cols=1, shared_xaxes=True)

    fig.add_trace(go.Scatter(x=distance, y=survey_df['SurveyMainData']['X1'][0][0].flatten(), line=dict(color="#0072BD", width=1.5), name='X1'), row=1, col=1)
    fig.add_trace(go.Scatter(x=distance, y=survey_df['SurveyMainData']['X2'][0][0].flatten(), line=dict(color="#D95319", width=1.5), name='X2'), row=1, col=1)
    fig.add_trace(go.Scatter(x=distance, y=survey_df['SurveyMainData']['X3'][0][0].flatten(), line=dict(color="#EDB120", width=1.5), name='X3'), row=1, col=1)

    fig.add_trace(go.Scatter(x=distance, y=survey_df['SurveyMainData']['Y1'][0][0].flatten(), line=dict(color="#0072BD", width=1.5), name='Y1'), row=2, col=1)
    fig.add_trace(go.Scatter(x=distance, y=survey_df['SurveyMainData']['Y2'][0][0].flatten(), line=dict(color="#D95319", width=1.5), name='Y2'), row=2, col=1)
    fig.add_trace(go.Scatter(x=distance, y=survey_df['SurveyMainData']['Y3'][0][0].flatten(), line=dict(color="#EDB120", width=1.5), name='Y3'), row=2, col=1)

    fig.add_trace(go.Scatter(x=distance, y=survey_df['SurveyMainData']['Z1'][0][0].flatten(), line=dict(color="#0072BD", width=1.5), name='Z1'), row=3, col=1)
    fig.add_trace(go.Scatter(x=distance, y=survey_df['SurveyMainData']['Z2'][0][0].flatten(), line=dict(color="#D95319", width=1.5), name='Z2'), row=3, col=1)
    fig.add_trace(go.Scatter(x=distance, y=survey_df['SurveyMainData']['Z3'][0][0].flatten(), line=dict(color="#EDB120", width=1.5), name='Z3'), row=3, col=1)

    fig.add_trace(go.Scatter(x=distance, y=survey_df['SurveyMainData']['X12'][0][0].flatten(), line=dict(color="#0072BD", width=1.5), name='X12'), row=4, col=1)
    fig.add_trace(go.Scatter(x=distance, y=survey_df['SurveyMainData']['X23'][0][0].flatten(), line=dict(color="#D95319", width=1.5), name='X23'), row=4, col=1)

    fig.add_trace(go.Scatter(x=distance, y=survey_df['SurveyMainData']['Y12'][0][0].flatten(), line=dict(color="#0072BD", width=1.5), name='Y12'), row=5, col=1)
    fig.add_trace(go.Scatter(x=distance, y=survey_df['SurveyMainData']['Y23'][0][0].flatten(), line=dict(color="#D95319", width=1.5), name='Y23'), row=5, col=1)

    fig.add_trace(go.Scatter(x=distance, y=survey_df['SurveyMainData']['Z12'][0][0].flatten(), line=dict(color="#0072BD", width=1.5), name='Z12'), row=6, col=1)
    fig.add_trace(go.Scatter(x=distance, y=survey_df['SurveyMainData']['Z23'][0][0].flatten(), line=dict(color="#D95319", width=1.5), name='Z23'), row=6, col=1)

    fig.add_trace(go.Scatter(x=distance, y=survey_df['SurveyMainData']['Xdd'][0][0].flatten(), line=dict(color="#0072BD", width=1.5), name='X12 - X23'), row=7, col=1)
    fig.add_trace(go.Scatter(x=distance, y=survey_df['SurveyMainData']['Ydd'][0][0].flatten(), line=dict(color="#D95319", width=1.5), name='Y12 - Y23'), row=7, col=1)
    fig.add_trace(go.Scatter(x=distance, y=survey_df['SurveyMainData']['Zdd'][0][0].flatten(), line=dict(color="#EDB120", width=1.5), name='Z12 - Z23'), row=7, col=1)

    fig.add_trace(go.Bar(x=anomaly_df.iloc[:, 1], y=anomaly_df.iloc[:, 7], width=1, text=anomaly_df.iloc[:, 0], textposition="outside", textfont=dict(size=100,color="black"), marker=dict(color = list(map(SetColor, anomaly_df.iloc[:, 7])))), row=8, col=1)

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='black')
    fig.update_yaxes(fixedrange=False)

    fig.update_layout(margin=dict(l=10, r=10, t=40, b=20))

    config = dict({'scrollZoom': True})

    pio.renderers.default = "browser"

    fig.show(config=config)
    
#SCZDeletion()
#listmatcher()
#anomaly_df_deleted = anomaly_df[anomaly_df['Delete'].isna()]
plotStripChart(anomaly_df)

# saving processed appendix C
#anomaly_df.to_excel('C:/Users/New/Downloads/Projects/Estonia/22-AN-EST-ROS-ELR-01/4 Data Analysis/Original Project/D2 P2/Results/appendixC processed.xlsx', sheet_name='Anomaly Table', index = False)


