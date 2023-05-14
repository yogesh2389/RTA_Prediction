# -*- coding: utf-8 -*-
"""
Created on Sun May 14 09:31:54 2023

@author: Yogesh
"""

import streamlit as st

# Define a function to create the UI
def app():
    st.title("My Streamlit App")
    name = st.text_input("Enter your name:")
    age = st.number_input("Enter your age:", min_value=0, max_value=150, step=1)
    email = st.text_input("Enter your email:")
    st.write("Hello,", name)
    st.write("You are", age, "years old")
    st.write("Your email address is", email)

# Run the app
if __name__ == '__main__':
    app()