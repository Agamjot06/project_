import pandas as pd
import streamlit as st
import joblib
    
model = joblib.load('xgboost_model.joblib')

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

# Sidebar
st.sidebar.header("Sidebar")
st.sidebar.write("Select a page to navigate:")
page= st.sidebar.selectbox("sidebar selectbox",["Page 1","Page 2","Page 3"])

#multi page application
if page == "Page 1":
    st.subheader("Introduction to My Project")
    st.image("https://img.freepik.com/free-photo/realistic-water-drop-with-ecosystem_23-2151196394.jpg", use_container_width=True)
    st.write("This project aims to predict the potability of water based on various water quality parameters. The model is trained using a dataset that includes features such as pH, hardness, solids, chloramines, sulfate, conductivity, organic carbon, trihalomethanes, and turbidity. The output will indicate whether the water is potable (safe for drinking) or non-potable (not safe for drinking).")   
    st.subheader("Let's discuss the parameters used in this project")
    st.write("1. **pH**: A measure of how acidic/basic water is. The range is from 0 to 14, with 7 being neutral.")
    st.write("2. **Hardness**: A measure of the concentration of calcium and magnesium in water, typically expressed in mg/L.")
    st.write("3. **Solids**: The total dissolved solids in water, measured in mg/L.")
    st.write("4. **Chloramines**: A disinfectant used in water treatment, measured in mg/L.")
    st.write("5. **Sulfate**: A naturally occurring mineral in water, measured in mg/L.")
    st.write("6. **Conductivity**: A measure of water's ability to conduct electricity, which is related to the concentration of ions in the water, measured in µS/cm (microsiemens per centimeter).")
    st.write("7. **Organic Carbon**: A measure of the amount of organic matter in water, which can affect water quality, measured in mg/L.")
    st.write("8. **Trihalomethanes**: Chemical compounds that can form when chlorine is used to disinfect water, measured in µg/L.")
    st.write("9. **Turbidity**: A measure of the cloudiness or haziness of water caused by large numbers of individual particles, measured in NTU (Nephelometric Turbidity Units).")
    

if page == "Page 2":
    col1,col2,col3 = st.columns(3)
    with col1:
        st.write("Enter the following Water Quality Parameters")
        ph = st.number_input("PH", 0, 14)
        chloramines = st.number_input("Chloramines(mg/L)", 0, 15)
        solids = st.number_input("Solids(mg/L)", 100, 1000)
    with col2:
        st.write("Enter the following Water Quality Parameters")
        Hardness = st.number_input("Hardness(mg/L)", 0, 1000)
        Sulfate = st.number_input("Sulfate(mg/L)", 0, 1000)
        Conductivity = st.number_input("Conductivity(µS/cm)", 0, 10000)
    with col3:
        st.write("Enter the following Water Quality Parameters")
        Organic_Carbon = st.number_input("Organic_Carbon(mg/L)", 0, 1000)
        Trihalomethanes = st.number_input("Trihalomethanes(µg/L)", 0, 1000)
        Turbidity = st.number_input("Turbidity(NTU)", 0, 100)
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
        final_pred = model.predict(new_data)
        st.header("Result: ")
        if final_pred == 1:
            st.image("https://nutritionsource.hsph.harvard.edu/wp-content/uploads/2024/11/AdobeStock_362493206.jpeg",use_container_width=True)
            st.subheader("Potable")
            st.success("Water is safe for Drinking.")
        else:
            st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQAP_ysr2jgFwOUye0ujpLUD8QkjAKnwjbvoA&s",use_container_width=True)
            st.subheader("Non-Potable")
            st.warning("Water is not safe for Drinking.")

if page == "Page 3":
    st.subheader("About the Project")
    st.write("This project is developed by [Your Name]. It aims to provide a user-friendly interface for predicting water potability based on various water quality parameters. The model is built using XGBoost, a powerful machine learning algorithm known for its performance and speed.")
    st.write("For more information, you can contact me at [Your Email].")
    st.image("https://www.example.com/your_image.jpg", use_container_width=True)  # Replace with your image URL