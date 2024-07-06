import streamlit as st
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler


# Loading the Models
model=pickle.load(open('LogisticRegression.pkl','rb'))
sc=pickle.load(open('Scaler.pkl','rb'))


# Prediction Function
def predict_results(input_data):
    input_data=np.array(input_data)
    input_data_reshaped=input_data.reshape(1,-1)

    data_scaled=sc.transform(input_data_reshaped)
    prediction=model.predict(data_scaled)

    print(prediction)

    if prediction[0]==0:
        return "Zero Insurance"
    else:
        return "Insurance Granted."



def main():

    # Title of the Page
    st.title("Insurance Classification Project 📝")

    st.info("Kaggle Competetion - Playground Series - Season 4, Episode 7")

    
    # Getting input data from user
    gender=st.text_input('Gender')
    age=st.text_input('Age')
    driving_license=st.text_input('Driving License')
    region_Code=st.text_input('Region Code')
    previously_Insured=st.text_input('Previously Insured')
    vehicle_age=st.text_input('Vehicle Age')
    vehicle_damage=st.text_input('Vehicle Damage')
    annual_premium=st.text_input('Annual Premium')
    policy_sales_channel=st.text_input('Policy Sales Channel')
    vintage=st.text_input('Vintage')


    # Code for prediction
    results=''


    # Creating button for Prediction
    if st.button("Insurence"):
        results=predict_results([gender,age,driving_license,region_Code,
                               previously_Insured,vehicle_age,vehicle_damage,
                                annual_premium, policy_sales_channel,vintage])
        
        st.success(results)




if  __name__=='__main__':
    main()