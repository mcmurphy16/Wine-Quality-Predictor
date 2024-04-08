#https://stackoverflow.com/questions/60866205/python-streamlit-run-issue
#was used to run the streamlit webpage
#use python -m streamlit run webpage-RF.py in command prompt of windows

#https://ngugijoan.medium.com/deploy-machine-learning-web-applications-with-streamlit-238b0380679d
#was used as a reference to build the webpage

import streamlit as st
import pandas as pd
import numpy as np
import pickle

from PIL import Image

#two pages will be there, one for white-wine and other for red-wine
app_mode = st.sidebar.selectbox('Select Page',['Red Wine','White Wine'])

#adding elements to red-wine page
if app_mode=='Red Wine': 
    st.title('Red Wine Quality Predictor')

    #adding images
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image('red-wine-1.jpg', width = 150)
    with col2:
        st.image('redwine3.jpg', width = 150)
    with col3:
        st.image('red-wine-2.jpg', width = 150)
    with col4:
        st.image('redwine4.jpg', width = 150)

    st.subheader('Set the parameters ')

    #adding sliders to set/adjust the features
    Alcohol = st.slider('Alcohol', 8.00, 15.00, 10.42)
    Residual_Sugar = st.slider('Residual Sugar', 0.00, 16.00, 2.54)
    Citric_Acid = st.slider('Citric Acid', 0.00, 1.00, 0.27)
    Density = st.slider('Density', 0.90, 1.20, 0.99)
    pH = st.slider('pH', 2.00, 5.00, 3.31)
    Volatile_Acidity = st.slider('Volatile Acidity', 0.00, 2.00, 0.53)
    Fixed_Acidity = st.slider('Fixed Acidity', 1.00, 15.00, 8.31)
    Sulphates = st.slider('Sulphates', 0.00, 2.00, 0.66)
    Free_Sulfurdioxide = st.slider('Free Sulfurdioxide', 0.00, 80.00, 15.87)
    Total_Sulfurdioxide = st.slider('Total Sulfurdioxide', 0.00, 300.00, 46.47)
    Chlorides = st.slider('Chlorides', 0.00, 1.00, 0.08)


    subdata={
        'Alcohol' : Alcohol,
        'Residual Sugar' : Residual_Sugar,
        'Citric Acid' : Citric_Acid,
        'Density' : Density,
        'pH' : pH,
        'Volatile Acidity' : Volatile_Acidity,
        'Fixed Acidity' : Fixed_Acidity,
        'Sulphates' : Sulphates,
        'Free Sulfurdioxide' : Free_Sulfurdioxide,
        'Total Sulfurdioxide' : Total_Sulfurdioxide,
        'Chlorides' : Chlorides
        }
    
    features = [Fixed_Acidity, Volatile_Acidity, Citric_Acid, Residual_Sugar, Chlorides,
                Free_Sulfurdioxide, Total_Sulfurdioxide, Density, pH, Sulphates, Alcohol]
    
    results = np.array(features).reshape(1, -1)


    #creating the predict button
    if st.button("Predict"):
        picklefile = open("redwinequality.pkl", "rb")
        model = pickle.load(picklefile)
        prediction = model.predict(results)
        st.write('Quality ', prediction[0], 'or (', np.round(prediction[0]), ')')
    
        



#adding elements to white-wine page 
elif app_mode=='White Wine':
    st.title('White Wine Quality Predictor')
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image('white-wine-1.jpg', width = 150)
    with col2:
        st.image('whitewine3.jpg', width = 150)
    with col3:
        st.image('white-wine-2.jpg', width = 150)
    with col4:
        st.image('whitewine4.jpg', width = 150)
    st.subheader('Set the parameters ')

    Alcohol = st.slider('Alcohol', 8.00, 15.00, 10.51)
    Residual_Sugar = st.slider('Residual Sugar', 0.00, 70.00, 6.39)
    Citric_Acid = st.slider('Citric Acid', 0.00, 2.00, 0.33)
    Density = st.slider('Density', 0.90, 1.20, 0.99)
    pH = st.slider('pH', 2.00, 5.00, 3.19)
    Volatile_Acidity = st.slider('Volatile Acidity', 0.00, 2.00, 0.28)
    Fixed_Acidity = st.slider('Fixed Acidity', 1.00, 15.00, 6.85)
    Sulphates = st.slider('Sulphates', 0.00, 2.00, 0.49)
    Free_Sulfurdioxide = st.slider('Free Sulfurdioxide', 0.00, 300.00, 35.30)
    Total_Sulfurdioxide = st.slider('Total Sulfurdioxide', 0.00, 500.00, 138.36)
    Chlorides = st.slider('Chlorides', 0.00, 1.00, 0.04)


    subdata={
        'Alcohol' : Alcohol,
        'Residual Sugar' : Residual_Sugar,
        'Citric Acid' : Citric_Acid,
        'Density' : Density,
        'pH' : pH,
        'Volatile Acidity' : Volatile_Acidity,
        'Fixed Acidity' : Fixed_Acidity,
        'Sulphates' : Sulphates,
        'Free Sulfurdioxide' : Free_Sulfurdioxide,
        'Total Sulfurdioxide' : Total_Sulfurdioxide,
        'Chlorides' : Chlorides
        }
    
    features = [Fixed_Acidity, Volatile_Acidity, Citric_Acid, Residual_Sugar, Chlorides,
                Free_Sulfurdioxide, Total_Sulfurdioxide, Density, pH, Sulphates, Alcohol]
    
    results = np.array(features).reshape(1, -1)


    #creating the predict button
    if st.button("Predict"):
        picklefile = open("whitewinequality.pkl", "rb")
        model = pickle.load(picklefile)
        prediction = model.predict(results)
        st.write('Quality ', prediction[0], 'or (', np.round(prediction[0]), ')')

