import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).with_name("mystreamlitappmodel.sv")

# Integer codes match the columns stored on the pickled forest.
SEX = {0: "Woman", 1: "Man"}
CHEST_PAIN = {
    0: "ATA",
    1: "NAP",
    2: "ASY",
    3: "TA",
}
RESTING_ECG = {0: "Normal", 1: "ST", 2: "LVH"}
EXERCISE_ANGINA = {0: "No", 1: "Yes"}
ST_SLOPE = {0: "No", 1: "Yes"}

FEATURE_COLUMNS = [
    "Age",
    "Sex",
    "ChestPainType",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "RestingECG",
    "MaxHR",
    "ExerciseAngina",
    "Oldpeak",
    "ST_Slope",
]


@st.cache_resource
def load_model():
    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)
    trained_columns = [str(name) for name in model.feature_names_in_]
    if trained_columns != FEATURE_COLUMNS:
        raise RuntimeError(
            "Saved model features do not match the form: "
            + ", ".join(trained_columns)
        )
    return model


def main():
    st.set_page_config(
        page_title="App to predict whether a person will develop a heart disease"
    )
    model = load_model()

    st.image(
        "https://cdn.pixabay.com/photo/2015/04/24/22/19/heart-738385_960_720.png",
        width=280,
    )
    st.title("App to predict whether a person will develop a heart disease")
    st.caption(
        "Estimate from a saved model. It is not a medical diagnosis."
    )

    left, right = st.columns(2)

    with left:
        sex = st.radio("Gender", list(SEX), format_func=lambda value: SEX[value])
        chest_pain = st.radio(
            "Chest Pain Type",
            list(CHEST_PAIN),
            format_func=lambda value: CHEST_PAIN[value],
            help="ATA atypical angina, NAP non-anginal pain, ASY asymptomatic, TA typical angina.",
        )
        resting_ecg = st.radio(
            "Resting ECG",
            list(RESTING_ECG),
            index=2,
            format_func=lambda value: RESTING_ECG[value],
        )
        exercise_angina = st.radio(
            "Exercise Induced Angina",
            list(EXERCISE_ANGINA),
            format_func=lambda value: EXERCISE_ANGINA[value],
        )
        st_slope = st.radio(
            "Slope of the peak exercise ST segment",
            list(ST_SLOPE),
            format_func=lambda value: ST_SLOPE[value],
        )

    with right:
        age = st.slider("Age", min_value=28, max_value=80, value=55)
        # RestingBP, cholesterol, fasting blood sugar, and oldpeak were stored
        # as 0/1 flags on the trained forest. A wider slider would collapse to
        # the same split and look more precise than the model is.
        resting_bp = st.slider("RestingBP", min_value=0, max_value=1, value=0)
        cholesterol = st.slider("Cholesterol", min_value=0, max_value=1, value=0)
        fasting_bs = st.slider("FastingBS", min_value=0, max_value=1, value=0)
        max_hr = st.slider("MaxHR", min_value=0, max_value=200, value=80, step=1)
        oldpeak = st.slider("Oldpeak", min_value=0, max_value=1, value=0)

    features = pd.DataFrame(
        [[
            age,
            sex,
            chest_pain,
            resting_bp,
            cholesterol,
            fasting_bs,
            resting_ecg,
            max_hr,
            exercise_angina,
            oldpeak,
            st_slope,
        ]],
        columns=FEATURE_COLUMNS,
    )
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0, prediction])

    st.subheader("Would I develop a heart disease?")
    st.subheader("Yes" if prediction == 1 else "No")
    st.write("Probability {0:.2f} %".format(probability * 100))


if __name__ == "__main__":
    main()
