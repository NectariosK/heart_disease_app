# źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic)

import streamlit as st
import pickle
from datetime import datetime
startTime = datetime.now()
# import znanych nam bibliotek

filename = "mystreamlitappmodel.sv"
mystreamlitappmodel = pickle.load(open(filename,'rb'))
# otwieramy wcześniej wytrenowany model

sex_d = {0:"Woman",1:"Man"}
#pclass_d = {0:"First",1:"Second", 2:"Third"}
chestPainType_d = {0:"ATA",1:"NAP", 2:"ABY", 3: "ASY"}
restingECG_d = {0:"Normal", 1:"ST", 2:"LVH"}
exerciseAngina_d = {0:"No", 1:"Yes"}
stslope_d = {0:"No", 1:"Yes"}
heartdisease_d = {0: "Normal", 1: "Fixed defect", 2 : "Reversable defect"}
# o ile wcześniej kodowaliśmy nasze zmienne, to teraz wprowadzamy etykiety z ich nazewnictwem

def main():
    st.set_page_config(page_title="App to predict whether a person will develop a heart disease")
    overview = st.container()
    left, right = st.columns(2)
    prediction = st.container()
    
    st.image("https://cdn.pixabay.com/photo/2015/04/24/22/19/heart-738385_960_720.png")
    
    with overview:
        st.title("App to predict whether a person will develop a heart disease")   

    with left:
        sex_radio = st.radio( "Gender", list(sex_d.keys()), format_func=lambda x : sex_d[x] )
        chestPainType_radio = st.radio( "Chest Pain Type", list(chestPainType_d.keys()), format_func=lambda x: chestPainType_d[x])
        restingECG_radio = st.radio( "Resting ECG", list(restingECG_d.keys()), index=2, format_func= lambda x: restingECG_d[x])
        exerciseAngina_radio = st.radio( "Exercise Induced Angina", list(exerciseAngina_d.keys()), format_func=lambda x: exerciseAngina_d[x])
        stslope_radio = st.radio( "Slope of the peak exercise ST segment", list(stslope_d.keys()), format_func=lambda x: stslope_d[x])
        heartdisease_radio = st.radio( "Heart disease", list(heartdisease_d.keys()), index=2, format_func= lambda x: heartdisease_d[x])
	
    with right:
        age_slider = st.slider("Age", value=1, min_value=1, max_value=80)
        cholesterol_slider = st.slider("Cholesterol", min_value=0, max_value=350)
        maxHR_slider = st.slider("MaxHR", min_value=0, max_value=500, step=1)
        oldpeak_slider = st.slider("Oldpeak", min_value=0, max_value=3, step=1)
        restingBP_slider = st.slider("RestingBP", min_value=0, max_value=1)
        fastingBS_slider = st.slider("FastingBS", min_value=0, max_value=1)

        data = [[sex_radio, chestPainType_radio, restingBP_slider, cholesterol_slider, 
        fastingBS_slider, restingECG_radio, maxHR_slider, exerciseAngina_radio, oldpeak_slider, stslope_radio, heartdisease_radio]]
        develop_heart_disease = mystreamlitappmodel.predict(data)
        s_confidence = mystreamlitappmodel.predict_proba(data)
    
    with prediction:
        st.subheader("Would I develop a heart disease?")
        st.subheader(("Yes" if develop_heart_disease[0] == 1 else "No"))
        st.write("Probability {0:.2f} %".format(s_confidence[0][develop_heart_disease][0] * 100))

if __name__ == "__main__":
    main()
