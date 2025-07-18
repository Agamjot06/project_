import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
import joblib
    
model = joblib.load('xgboost_model.joblib')

page_element="""
<style>
[data-testid="stAppViewContainer"]{
  background-image: url("https://png.pngtree.com/thumb_back/fh260/background/20200606/pngtree-dual-color-gradient-with-rounded-light-effect-image_338148.jpg");
  background-size: cover;
}
[data-testid="stHeader"]{
  background-color: rgba(0,0,0,0);
}
</style>
"""
st.markdown(page_element, unsafe_allow_html=True)

st.title("Water Quality Prediction")
st.subheader("Here, We will predict whether the water is Drinkable or not!")

# Sidebar
st.sidebar.header("Sidebar")
st.sidebar.write("Select a page to navigate:")
page= st.sidebar.selectbox("sidebar selectbox",["Introduction","Prediction","About Dataset"])

#multi page application
if page == "Introduction":
    st.image("https://img.freepik.com/free-photo/realistic-water-drop-with-ecosystem_23-2151196394.jpg", use_container_width=True)
    st.subheader("Introduction to My Project")
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
    

if page == "Prediction":
    st.image("https://intownplumbingtx.com/wp-content/uploads/2024/01/water-quality-blog-post-img.jpg", use_container_width=True)
    st.write("Enter the water quality parameters below to predict its potability:")
    col1,col2,col3 = st.columns(3)
    with col1:
        ph = st.number_input("PH", 0, 14)
        chloramines = st.number_input("Chloramines(mg/L)", 0, 15)
        solids = st.number_input("Solids(mg/L)", 100, 6000)
    with col2:
        Hardness = st.number_input("Hardness(mg/L)", 0, 1000)
        Sulfate = st.number_input("Sulfate(mg/L)", 0, 1000)
        Conductivity = st.number_input("Conductivity(µS/cm)", 0, 10000)
    with col3:
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
            st.subheader("Potable")
            st.success("Water is safe for Drinking.")
            st.image("https://nutritionsource.hsph.harvard.edu/wp-content/uploads/2024/11/AdobeStock_362493206.jpeg",use_container_width=True)
        else:
            st.subheader("Non-Potable")
            st.warning("Water is not safe for Drinking.")
            st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQAP_ysr2jgFwOUye0ujpLUD8QkjAKnwjbvoA&s",use_container_width=True)

if page == "About Dataset":
    st.subheader("Here are some graphs to visualize the data")
    df = pd.read_csv("water_potability dataset.csv")
    st.write("Dataset Shape: ", df.shape)
    st.write("Dataset Columns: ", df.columns.tolist())
    st.write("There are some null values in the dataset, which will be handled during model training. Null values are present in the columns 'ph', 'Sulfate', 'Trihalomethanes'")
    st. write(" Let's visualize the distribution of various columns:")
    selected_column = st.selectbox(
    "Select columns to visualize",
    ["ph", "Sulfate", "Trihalomethanes", "Solids", "Hardness", 
     "Chloramines", "Conductivity", "Organic_Carbon", "Turbidity"]
)

# Define labels for each column
    column_labels = {
        "ph": ("pH", "pH"),
        "Sulfate": ("Sulfate (mg/L)", "Sulfate"),
        "Trihalomethanes": ("Trihalomethanes", "Trihalomethanes"),
        "Solids": ("Solids (mg/L)", "Solids"),
        "Hardness": ("Hardness (mg/L)", "Hardness"),
        "Chloramines": ("Chloramines (mg/L)", "Chloramines"),
        "Conductivity": ("Conductivity (µS/cm)", "Conductivity"),
        "Organic_Carbon": ("Organic Carbon (mg/L)", "Organic Carbon"),
        "Turbidity": ("Turbidity (NTU)", "Turbidity"),
    }

    # Plot
    if selected_column:
        xlabel, title = column_labels[selected_column]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df[selected_column].dropna(), bins=30, alpha=0.5, color='darkgreen')
        ax.set_title(f"Distribution of {title}")
        ax.set_xlabel(xlabel)
        st.pyplot(fig)
    
    st.write("Correlation Heatmap")
    st.write("correlation between different features in the dataset can provide insights into how they relate to each other. Below is a heatmap showing the correlation between various water quality parameters:")
    correlation=df.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", ax=ax) 
    ax.set_title('Correlation Heatmap')
    st.pyplot(fig)
    
    st.write("Potability Distribution")
    st.write("The distribution of potability in the dataset can be visualized using a count plot. This will show how many samples are potable (1) and non-potable (0).")
    fig, ax = plt.subplots(figsize=(10, 6)) 
    sns.countplot(x='Potability', data=df, ax=ax, palette='Set2')
    ax.set_title('Potability Distribution')
    ax.set_xlabel('Potability')
    ax.set_ylabel('Count')
    st.pyplot(fig)
    st.write("The dataset contains a total of {} samples, with {} potable and {} non-potable samples.".format(
        df.shape[0], df['Potability'].value_counts()[1], df['Potability'].value_counts()[0]))
    

    
    st.subheader("About the Project")
    st.write("This project is developed by AGAMJOT KAUR, Btech(ECM) 3rd Year. It aims to provide a user-friendly interface for predicting water potability based on various water quality parameters. The model is built using XGBoost, a powerful machine learning algorithm known for its performance and speed.")
    