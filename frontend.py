import streamlit as st
import pandas as pd 
import pickle
st.set_page_config(page_title = 'Car price predictor')
st.header('Welcome , lets predict the car price')
df = pd.read_csv('copiedStr.csv')
with open('XGBoost.pkl','rb') as file:
    xgmodel=pickle.load(file) 
with st.container(border=True) :
    c1,c2 = st.columns(2)
    model = c1.selectbox('Model', options = df['Make'].unique())
    year = c2.number_input('Year', min_value= 2000)
    enginesize = c1.number_input('Engine Size', min_value=0.00, step = 0.1)
    mileage = c2.number_input('Mileage',min_value=5000, step = 1000)
    fuel = c1.radio('Fuel Type', options= df['Fuel Type'].unique())
    transmission = c2.radio('Transmission',options=df['Transmission'].unique())
    carmodels = list(df['Make'].unique())
    carmodels.sort()
    fueltypes = list(df['Fuel Type'].unique())
    fueltypes.sort()
    transmissions = list(df['Transmission'].unique())
    transmissions.sort()
    inputs =[[carmodels.index(model), year,enginesize,mileage,fueltypes.index(fuel),transmissions.index(transmission)]]
       
    if c1.button('Predict price'):
        df = df.drop(['Unnamed: 0'], axis = 1)
        out= xgmodel.predict(inputs)
    
        st.subheader(f'Car price💰: {out[0]*1000000}')
