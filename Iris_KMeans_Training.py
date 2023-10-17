import streamlit as st

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score


st.title("Prediction IRIS Datasets")

data = load_iris()

df = pd.DataFrame(data=data.data, columns=data.feature_names)
df["Species"] = data.target
df["Species_Name"] = df["Species"]
for index, row in df.iterrows():
    species = int(row["Species"])
    df.loc[index, "Species_Name"] = data.target_names[species]

st.subheader('Sample of IRIS dataset')
st.write(df.sample(5))

st.divider()

tab1, tab2, tab3 = st.tabs(["3D Scatter Plot", "Box Plot", "Violin Plot"])

with tab1:
    st.subheader('3D Scatter Plot of IRIS dataset')
    fig1 = px.scatter_3d(df,
                    x="petal length (cm)",
                    y="petal width (cm)",
                    z="sepal length (cm)",
                    color="Species_Name",
                    height=500)
    st.plotly_chart(fig1, use_container_width=True)
    
with tab2:
    st.subheader('Box Plot of IRIS dataset')
    fig2 = px.box(df, x=['sepal length (cm)', 'sepal width (cm)',
                        'petal length (cm)', 'petal width (cm)'],
                 color="Species_Name")
    
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader('Violin Plot of IRIS dataset')
    fig3 = px.violin(df, y=['sepal length (cm)', 'sepal width (cm)',
                    'petal length (cm)', 'petal width (cm)'],
                color="Species_Name",
                points='all', box=False)
    
    st.plotly_chart(fig3, use_container_width=True)

st.divider()
x = df.iloc[:,:4].copy()
st.write()
st.subheader('K-Means Cluster Model')

st.sidebar.subheader('K-Means Cluster Model parameters')
N_cluster = st.sidebar.slider('How many cluster you want?', 0, 20, 3)
Random_state = st.sidebar.slider('Select the Random State?', 0, 42, 0)
Algorithm_selected = st.sidebar.selectbox(
    'Wich algorithm you want to select?',
    ("lloyd", "elkan", "auto", "full"))


kmeans = KMeans(n_clusters=N_cluster, random_state=Random_state, n_init="auto",algorithm = Algorithm_selected)

kmeans.fit(x)
x['species'] = kmeans.labels_
x['species'] = x['species'].astype('category')

st.write()
st.subheader('3D ScatterPlot of Clustered IRIS dataset ')
fig4 = px.scatter_3d(x,
                    x="petal length (cm)",
                    y="petal width (cm)",
                    z="sepal length (cm)",
                    color="species",
                    height=500)

st.plotly_chart(fig4, use_container_width=True)
st.caption("Note: We can  only compare this graphic with the original one when We use 3 as the  number of clusters for our model and then see how well the model is performing!!")


x['species'] = np.where(x['species'] == 2,'virginica', np.where(x['species'] == 0, 'versicolor', 'setosa' ))
df['cluster'] = x['species']

st.divider()
st.subheader("Model Evaluation in Training")
if N_cluster == 3:
    cm = confusion_matrix(df['Species_Name'], df['cluster'], labels=data.target_names)

    # Streamlit app
    st.title('Confusion Matrix Display')

    # Convert the confusion matrix plot to an image
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g',cmap='Blues', ax = ax)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    st.pyplot(fig,)

    



#st.write(f"The Model score is: {-(kmeans.score((x.iloc[:,:4])))}")
st.success(f"The Model score is: {-(kmeans.score((x.iloc[:,:4])))}", icon='👍')


pickle.dump(kmeans, open("pages/streamlit_objects/kmeans.pkl", "wb"))


