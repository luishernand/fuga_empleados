#--------------#
#   Libraries #
#--------------#
import streamlit as st
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import OneHotEncoder


st.set_option('deprecation.showPyplotGlobalUse', False)
#st.set_page_config(layout="wide")

st.sidebar.warning('Este es un proyecto de prueba')
#--------------#
#   dataset    #
#--------------#

df = pd.read_csv('C:\\Users\\User\\OneDrive\\Documentos\\Cursos de ML\\Data Science 4B\\Primer dia en la empresa\\AbandonoEmpleados.csv', sep= ';',index_col='id', na_values='#N/D')
df_preprocess = pd.read_csv('C:\\Users\\User\\OneDrive\\Documentos\\Cursos de ML\\Data Science 4B\\Primer dia en la empresa\\dataforml.csv', index_col=[0])

#-------------------#
#   Carga de modelo #
#-------------------#
modelo = joblib.load('C:\\Users\\User\\OneDrive\\Documentos\\Cursos de ML\\Data Science 4B\\Primer dia en la empresa\\trained_model.jl')


#--------------#
#   Body       #
#--------------#
st.header('Dashboard fuga de empleados')



#st.write(df_preprocess.columns)

#--------------#
#  Predictions #
#--------------#


df['score_abandono_new'] = modelo.predict_proba(df_preprocess.drop(columns='abandono'))[:, 1]

df = df.sort_values(by ='score_abandono_new', ascending =False)

#----------------------#
#  Columnas y metricas #
#----------------------#
col1,col2,col3= st.columns(3)
#tasa de fuga
tasa_fuga = round(df_preprocess['abandono'].sum() / df_preprocess['abandono'].count() *100,2)
col1.metric('Tasa de Fuga', value = str(tasa_fuga) + '%')

#empleados en riesgo
en_riesgo = np.where(df['score_abandono_new']>0.5, 1,0)
en_riesgo = en_riesgo.sum()
col2.metric('Empleados En Riesgo', value = en_riesgo)

#Calculamos el impacto economico de las empresa
#creamos nueva variable salario_ano empleados
df['salario_ano'] = df['salario_mes'] * 12

# listar las condicones/ filtar por tipo de condicion el df
condiciones = [(df['salario_ano']<=30000),(df['salario_ano']>30000) & (df['salario_ano']<=50000),(df['salario_ano']>50000) & (df['salario_ano']<=75000),(df['salario_ano']>75000)]
resultados = [df.salario_ano *0.161, df.salario_ano * 0.197, df.salario_ano * 0.204, df.salario_ano * 0.21]

#aplicandolo las condiciones al resultado
df['impacto_abandono'] = np.select(condiciones, resultados, default=-999)
col3.metric('Impacto Economico', value =round(df['impacto_abandono'].sum()) )

with st.expander('Riesgo por puesto'):
    fig = px.box(df, x="puesto", y="score_abandono_new")
    st.plotly_chart(fig)

st.write('##### Empleados con Mayor Riesgos')
registros = df.shape[0]
max_reg =len(df)

number = st.number_input("Entre la cantidad de Registro a Mostrar", min_value=1, max_value=max_reg)

st.write(df.head(number))


    