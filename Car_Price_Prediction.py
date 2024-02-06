import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from sklearn.preprocessing import StandardScaler

   
st.markdown(
    f"<h1 style='color:#F24C3D; font-size: 38px;'>Car Price Prediction</h1>",
    unsafe_allow_html=True,)
df=pd.read_csv(r"C:\Users\SKAN\Desktop\Raajee\cardekho\cardekho2.csv")
df.drop(["Unnamed: 0","Unnamed: 0.1"],axis=1,inplace=True)
col1,col2=st.columns(2)
with col1:
    model=st.selectbox(':blue[Select a Model]',df.model.unique())
    ft=st.text_input(':blue[Fuel_Type: Petrol: 0, Diesel: 1, Cng: 2, Lpg: 3, Electric: 4]')
    km=st.text_input(':blue[Kilo Meter(Float)]')  
with col2:    
    transmission=st.text_input(':blue[Transmission_type: Automatic: 0, Manual: 1]')
    ownerNo=st.text_input(':blue[Owener_NO]')
    modelYear=st.text_input(':blue[Model_Year]')

predict=st.button("Predict")

if predict:
    df1=pd.read_csv(r"C:\Users\SKAN\Desktop\Raajee\cardekho\cardekho1.csv")
    df1.drop(["Unnamed: 0","Unnamed: 0.1"],axis=1,inplace=True)    
    X=df1.drop("price[₹Lakh]",axis=1)
    

    def predict_price(model,ft,km,transmission,ownerNo,modelYear):
            x=np.zeros(len(X.columns)) 
            loc_index=np.where(X.columns==model)[0][0]  
            if loc_index>=0:
                x[loc_index]=1                         
            x[0]=ft
            x[1]=km
            x[2]=transmission
            x[3]=ownerNo
            x[4]=modelYear
            
            return x
    import pickle
    sc=StandardScaler()

    with open(r"C:\Users\SKAN\Desktop\Raajee\cardekho\model.pkl", 'rb') as file:
       plr= pickle.load(file)

    x=predict_price(str(model),ft,km,transmission,ownerNo,modelYear)
    result=plr.predict([x])[0]

    formatted_message = f'<span style="color:#00cc00; font-weight:bold;font-size: 30px">The Predicted price of the car is Rs. {result:.2f} Lakh</span>'
    st.markdown(formatted_message, unsafe_allow_html=True)
