import streamlit as st
import pandas as pd
st.header("Example")
url = "https://drive.google.com/file/d/1u5QRmPuDhlpfxqQYkS8w1AkxOk3BnwDo/view?usp=sharing"
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
@st.cache
def df_from_url(path):
    df = pd.read_csv(path)
    return df
st.write(df_from_url(path))