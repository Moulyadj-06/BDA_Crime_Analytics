from openai import OpenAI
import streamlit as st

def get_openai_client():
    api_key = st.secrets["OPENAI_API_KEY"]
    return OpenAI(api_key=api_key)
