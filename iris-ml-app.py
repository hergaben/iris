import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Простое приложение Для Определения Цветов Ириса
# Это приложение определяет тип ** ЦВЕТКА ИРИСА **!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Длина чашелистика', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Ширина чашелистика', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Длина лепестка', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Ширина лепестка', 0.1, 2.5, 0.2)
    data = {'Длина чашелистика': sepal_length,
            'Ширина чашелистика': sepal_width,
            'Длина лепестка': petal_length,
            'Ширина лепестка': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)