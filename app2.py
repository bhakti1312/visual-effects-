import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.ensemble import RandomForestClassifier


st.title("Human vs AI Comparison System")


file = st.file_uploader("Upload CSV", type=["csv"])


if file is not None:


    data = pd.read_csv(file)

    st.write(data.head())


    # =============================
    # Convert ALL text to numbers
    # =============================

    le = LabelEncoder()

    for column in data.columns:

        if data[column].dtype == object:

            data[column] = le.fit_transform(data[column])


    # =============================

    X = data.iloc[:,:-1]

    y = data.iloc[:,-1]


    # =============================
    # Scaling
    # =============================

    sc = StandardScaler()

    X = sc.fit_transform(X)


    # =============================

    X_train, X_test, y_train, y_test = train_test_split(

        X,y,test_size=0.2,random_state=42)


    model = RandomForestClassifier()

    model.fit(X_train,y_train)

    pred = model.predict(X_test)


    acc = accuracy_score(y_test,pred)

    pre = precision_score(y_test,pred,average="weighted")

    rec = recall_score(y_test,pred,average="weighted")

    f1 = f1_score(y_test,pred,average="weighted")


    st.write("Accuracy:",acc)

    st.write("Precision:",pre)

    st.write("Recall:",rec)

    st.write("F1 Score:",f1)
