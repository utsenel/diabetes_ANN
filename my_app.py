import streamlit as st

import pickle

import pandas as pd

from sklearn.preprocessing import StandardScaler


# cd ~/appdata/local/programs/python/python311/scripts

# ./streamlit run ~/Documents/Repo/diabetes-prediction/main.py

 

st.set_page_config(page_title='Diabetes Prediction')

 

st.write("Kişinin değerlerini girin")

 

form = st.form(key="Form1")

c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns(9)

 

with c1:

    Pregnancies = form.text_input("Pregnancies")

with c2:

    Glucose = form.text_input("Glucose")

with c3:

    BloodPressure = form.text_input("BloodPressure")

with c4:

    SkinThickness = form.text_input("SkinThickness")

with c5:

    Insulin = form.text_input("Insulin")

with c6:

    Bmi = form.text_input("BMI")

with c7:

    DiabetesPedigreeFunction = form.text_input("DiabetesPedigreeFunction")

with c8:

    Age = form.text_input("Age")

 

submitButton = form.form_submit_button(label = 'Predict')

 

# Scale

data = pd.read_csv("diabetes.csv")

X= data.drop(["Outcome"],axis=1)

 

scaler=StandardScaler()

scaler.fit_transform(X)

 

if submitButton:

    with st.spinner('Predicting....'):

        loaded_model = pickle.load(open('model.sav', 'rb'))

        input_frame=pd.DataFrame(scaler.transform([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, Bmi, DiabetesPedigreeFunction, Age]]),

                                        columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"])

        result = loaded_model.predict(input_frame)

 

        st.write('Kişinin Durumu: ')

 

        if result == 0:

            st.text('Bilgileri paylaşılan kişinin sağlıklı olduğu düşünülmektedir.')

        else:

            st.text('Bilgileri paylaşılan kişinin diyabet hastası olduğu düşünülmektedir.')