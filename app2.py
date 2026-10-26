"""Heart-disease estimate form backed by a saved random forest.

The forest was pickled with scikit-learn 1.0.2 and expects the eleven
columns in FEATURE_COLUMNS, in that order. Codes below are the integers
stored on the model, not a fresh encoding of the clinical labels.

Several columns were stored as 0/1 flags, so the matching controls only
offer those two values. Resting ECG is shown with three labels, but the
forest's only split for that column is at 0.5, so ST and LVH take the
same branch. The result is a model estimate, not a medical diagnosis.
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

# Kept next to this file so Community Cloud and a local run resolve the same path.
MODEL_PATH = Path(__file__).with_name("mystreamlitappmodel.sv")

# Integer codes match the columns stored on the pickled forest.
SEX = {0: "Woman", 1: "Man"}
CHEST_PAIN = {
    0: "ATA",  # atypical angina
    1: "NAP",  # non-anginal pain
    2: "ASY",  # asymptomatic
    3: "TA",  # typical angina
}
RESTING_ECG = {0: "Normal", 1: "ST", 2: "LVH"}
EXERCISE_ANGINA = {0: "No", 1: "Yes"}
# Clinical ST slope is usually Up/Flat/Down. This forest only splits the
# column at 0.5, so the form keeps the two values it was trained on.
ST_SLOPE = {0: "No", 1: "Yes"}

# Names and order come from the forest's feature_names_in_.
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
    """Load the forest once per server process and refuse a column mismatch.

    A mismatch used to fail silently: age was dropped and an extra field was
    sent in its place, so the important columns were scored as the wrong inputs.
    """
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
        # Age and MaxHR are the continuous columns the forest actually splits
        # across a range. The other numeric inputs were stored as 0/1 flags;
        # a wider slider would collapse to the same branch and look more
        # precise than the model is.
        age = st.slider("Age", min_value=28, max_value=80, value=55)
        resting_bp = st.slider("RestingBP", min_value=0, max_value=1, value=0)
        cholesterol = st.slider("Cholesterol", min_value=0, max_value=1, value=0)
        fasting_bs = st.slider("FastingBS", min_value=0, max_value=1, value=0)
        max_hr = st.slider("MaxHR", min_value=0, max_value=200, value=80, step=1)
        # Oldpeak is the ST depression induced by exercise relative to rest,
        # usually in millimetres. A larger drop means the heart is under more
        # stress during exercise. This forest only splits the column at 0.5,
        # so the control is a 0/1 flag rather than the measured depression.
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
    # predict_proba columns follow classes_ [0, 1]. Report the share of trees
    # that voted for the chosen class, not a fixed column.
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0, prediction])

    st.subheader("Would I develop a heart disease?")
    st.subheader("Yes" if prediction == 1 else "No")
    st.write("Probability {0:.2f} %".format(probability * 100))


if __name__ == "__main__":
    main()
