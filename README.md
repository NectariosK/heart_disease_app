# Heart disease app

A Streamlit form that estimates whether a person may develop heart disease. It scores one saved random forest and shows the majority class (`Yes` or `No`) plus the share of trees that voted for that class.

This is a statistical estimate from a saved model. It is not a medical diagnosis, and it must not be used to make health decisions.

**Live app:** https://nectariosk-heart-disease-app-app2-6z9sgj.streamlit.app

## What it does

1. Collect the eleven inputs the forest was trained on.
2. Pass them, in the trained column order, to `mystreamlitappmodel.sv`.
3. Show class `1` as **Yes** and class `0` as **No**.
4. Show the predicted class probability as a percentage.

The forest is a `RandomForestClassifier` pickled with scikit-learn 1.0.2. It has 20 trees, `random_state=0`, and classes `[0, 1]`. The training rows are not in this repository, so the model cannot be retrained from this project.

## Inputs

The form codes match the integers stored on the model.

| Column | Control | Values the model uses |
| --- | --- | --- |
| Age | Slider, 28–80 | Years. The trees split ages from 32 through 76. |
| Sex | Woman / Man | 0 / 1 |
| ChestPainType | ATA, NAP, ASY, TA | 0 atypical angina, 1 non-anginal pain, 2 asymptomatic, 3 typical angina |
| RestingBP | 0 or 1 | The only split is at 0.5. |
| Cholesterol | 0 or 1 | The only split is at 0.5. |
| FastingBS | 0 or 1 | The only split is at 0.5. |
| RestingECG | Normal, ST, LVH | 0, 1, 2. The only split is at 0.5, so ST and LVH take the same branch. |
| MaxHR | Slider, 0–200 | The trees split this column from about 2 through 118. |
| ExerciseAngina | No / Yes | 0 / 1 |
| Oldpeak | 0 or 1 | The only split is at 0.5. |
| ST_Slope | No / Yes | 0 / 1. The forest did not store the usual Up / Flat / Down slope codes. |

Age, chest-pain type, and max heart rate account for almost all of the forest's splits. The 0/1 fields change the score only when they cross 0.5.

## Run locally

Use Python 3.10. The pinned scikit-learn release does not load on NumPy 1.24 or newer, and Streamlit 1.66 or newer requires Python 3.11.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app2.py
```

Open the local URL Streamlit prints, usually http://localhost:8501.

## Deployment

The live app is hosted on Streamlit Community Cloud from `main`, with `app2.py` as the entry point. That environment uses Python 3.10.22. `requirements.txt` pins Streamlit 1.65.0, NumPy 1.23.5, and scikit-learn 1.0.2 so the install succeeds and the pickle still loads.

Community Cloud puts an idle app to sleep. Opening the link and choosing to wake it starts the app again.

## Project layout

| File | Purpose |
| --- | --- |
| `app2.py` | Streamlit form and prediction |
| `mystreamlitappmodel.sv` | Pickled random forest |
| `requirements.txt` | Versions known to load that forest on Python 3.10 |

## Limits

- The estimate is not a diagnosis and has no clinical validation in this repository.
- Resting blood pressure, cholesterol, fasting blood sugar, oldpeak, and ST slope are flags, not the measurements those names usually mean.
- Max heart rate values above about 118 fall on the same side of the highest split.
- The displayed percentage is the forest's vote for the predicted class. It is not a calibrated risk.
