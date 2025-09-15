import streamlit
import joblib

model = joblib.load("regression.joblib")
size = streamlit.number_input("Size (sqm)", 10, 1000, 50)
bedrooms = streamlit.number_input("Number of bedrooms", 1, 10, 2)
garden = streamlit.number_input("Has garden?", 0, 1, 0)

features = [[size, bedrooms, garden]]
prediction = model.predict(features)[0]
streamlit.write(f"Predicted price: {prediction}")