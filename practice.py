import streamlit as st
st.segmented_control("Model type", ["Decision Tree", "Random Forest", "SVM"])
st.toggle("Use Standard Scaler?")
st.slider("Number of Trees", min_value=10, max_value=500, step=10)