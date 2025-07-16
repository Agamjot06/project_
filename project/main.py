import pandas as pd
import numpy as np
import streamlit as st
import pickle



with open("../savedModels/xgboost_model.pkl", "rb") as file:
    xgb = pickle.load(file)

page_bg_img = '''
<style>
body {
background-image: url("https://media.hswstatic.com/eyJidWNrZXQiOiJjb250ZW50Lmhzd3N0YXRpYy5jb20iLCJrZXkiOiJnaWZcL3dhdGVyLXVwZGF0ZS5qcGciLCJlZGl0cyI6eyJyZXNpemUiOnsid2lkdGgiOiIxMjAwIn19fQ==");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)
st.title("Water Quality Prediction")
st.subheader("Here, We will predict whether the water is Drinkable or not!")
ph = st.slider("PH", 0, 14)
chloramines = st.slider("Chloramines", 0, 15)
solids = st.slider("Solids", 100, 1000)
Hardness = st.number_input("Hardness")
Sulfate = st.number_input("Sulfate")
Conductivity = st.number_input("Conductivity")
Organic_Carbon = st.number_input("Organic_Carbon")
Trihalomethanes = st.number_input("Trihalomethanes")
Turbidity = st.number_input("Turbidity")
new_data = pd.DataFrame(
    {
        "ph": [ph],
        "Hardness": [Hardness],
        "Solids": [solids],
        "Chloramines": [chloramines],
        "Sulfate": [Sulfate],
        "Conductivity": [Conductivity],
        "Organic_carbon": [Organic_Carbon],
        "Trihalomethanes": [Trihalomethanes],
        "Turbidity": [Turbidity],
    }
)
if st.button("Predict"):
    final_pred = xgb.predict(new_data)
    st.header("Result: ")
    if final_pred == 1:
        st.image("https://nutritionsource.hsph.harvard.edu/wp-content/uploads/2024/11/AdobeStock_362493206.jpeg",use_container_width=True)
        st.subheader("Potable")
        st.success("Water is safe for Drinking.")
    else:
        st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQAP_ysr2jgFwOUye0ujpLUD8QkjAKnwjbvoA&s",use_container_width=True)
        st.subheader("Non-Potable")
        st.warning("Water is not safe for Drinking.")
