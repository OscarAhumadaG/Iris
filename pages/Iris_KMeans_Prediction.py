import streamlit as st

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import plotly.express as px
import matplotlib.pyplot as plt
import pickle

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score


st.title("IRIS model Prediction")

sepal_length = st.sidebar.number_input("Please select a value for the sepal lenght ", min_value=4.3, max_value=7.9, step = 0.1)
sepal_width = st.sidebar.number_input("Please select a value for the sepal width ", min_value=2.0, max_value=4.4 , step= 0.1)
petal_length = st.sidebar.number_input("Please select a value for the petal lenght", min_value=1.0, max_value=6.9, step = 0.1)
petal_width =st.sidebar.number_input("Please select a value for the petal width ", min_value=0.1, max_value=2.5, step = 0.1)


btn_classify = st.sidebar.button("Classify")

if btn_classify:
    st.write("Button was pressed")
    model =  pickle.load(open("pages/streamlit_objects/kmeans.pkl","rb"))
    df = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                     columns= ["sepal length (cm)","sepal width (cm)", "petal length (cm)" , "petal width (cm)" ])
    
    
    st.table(df)
    st.write("")
    prediction = model.predict(df)
    
    
    st.write(f"Cluster prediction: {prediction[0]}")
    num_clusters = len(model.cluster_centers_)
    if num_clusters == 3:
        if prediction[0] == 2:
            Type = 'virginica'
        elif prediction[0] == 0:
            Type ='versicolor'
        else:
            Type = 'setosa'
        st.write(f"Flower type:  {Type}")
    
    st.success("We've completed our prediction using K-Means!", icon="✅")
    







