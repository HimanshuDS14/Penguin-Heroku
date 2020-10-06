import streamlit as st
import pandas as pd
import pickle 
import numpy as np
import matplotlib.pyplot as plt





st.write("""  
         

# Penguin Prediction App

This app predicts the *** Palmer Penguins *** species



""")


st.sidebar.header("User Input Features")

st.sidebar.markdown("""
                    
[Example CSV input File] (https://www.kaggle.com/mainakchaudhuri/penguin-data-set)


""")


def user_input_feature():
    
    island = st.sidebar.selectbox('Island' , ('Biscoe' , 'Dream' , 'Torgersen'))
    sex = st.sidebar.selectbox('Sex' , ('male' , 'female'))
    bill_length_mm = st.sidebar.slider('Bill Length (mm)' , 32.1 , 53.6 , 43.9)
    bill_depth_mm = st.sidebar.slider("Bill Depth (mm)" , 13.1 , 21.5 , 17.2)
    flipper_length_mm = st.sidebar.slider("Filpper Length (mm)" , 172.0 , 231.0 , 201.0)
    body_mass_g = st.sidebar.slider("Body Mass (gm)" , 2700.0 , 6300.0 , 4207.0)
    
    
    data = {
            
            'island' : island , 
            'bill_length_mm' : bill_length_mm , 
            'bill_depth_mm' : bill_depth_mm , 
            'flipper_length_mm' : flipper_length_mm,
            'body_mass_g' : body_mass_g , 
            'sex' : sex
            
            }
    
    features = pd.DataFrame(data , index = [0])
    return features


f = user_input_feature()


st.write("# User Inputs")
st.write(f)

raw = pd.read_csv("Penguins_cleaned.csv")
penguins = raw.drop(columns = ['species'])
df = pd.concat([f , penguins] , axis=  0)


encode = ['sex' , 'island']

for col in encode:
    dummy = pd.get_dummies(df[col] , prefix = col)
    df = pd.concat([df , dummy] , axis=1)
    del df[col]
    
df = df[:1]


model = pickle.load(open("Penguin_model_01.pkl" , 'rb'))

pred = model.predict(df)
pred_prob = model.predict_proba(df)

st.subheader("Prediction")
label = np.array(['Adelie' , 'Chinstrap' , 'Gentoo'])
st.write(label[pred])

st.subheader("prediction Probability")
st.write(pred_prob)



 





















        