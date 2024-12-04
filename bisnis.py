from streamlit_option_menu import option_menu
import joblib
import streamlit as st
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import warnings
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Sentimen Analysis",
    page_icon='https://cdn-icons-png.flaticon.com/512/1998/1998664.png',
    layout='centered',
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)
st.write("""
<center><h2 style = "text-align: justify;">Analisi Sentimen Wisata Madura Dengan Maximum Entropy</h2></center>
""",unsafe_allow_html=True)

with st.container():
    with st.sidebar:
        selected = option_menu(
        st.write("""<h3 style = "text-align: center;"><img src="https://cdn-icons-png.flaticon.com/512/1998/1998664.png" width="120" height="120"></h3>""",unsafe_allow_html=True), 
        ["Home", "Dataset", "Implementation"], 
            icons=['house', 'bar-chart','check2-square', 'person'], menu_icon="cast", default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#412a7a"},
                "icon": {"color": "white", "font-size": "18px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "color":"white"},
                "nav-link-selected":{"background-color": "#412a7a"}
            }
        )
    if selected == "Home" :
        st.write("""<h3 style="text-align: center;">
        <img src="https://www.mongabay.co.id/wp-content/uploads/2019/08/PETANI-GARAM-LAMONGAN-4.jpg" width="500" height="300">
        </h3>""", unsafe_allow_html=True)
    if selected == "Dataset":
        file_path = 'data stopword tes.csv'  # Ganti dengan path ke file Anda
        data = pd.read_csv(file_path)
        st.write(data.head(10))
    if selected == "Implementation":
        import joblib
        # Menggunakan pandas untuk membaca file CSV
        file_path = 'data stopword tes.csv'  # Ganti dengan path ke file Anda
        data = pd.read_csv(file_path)
        vectorizer = TfidfVectorizer(max_features=100)
        X = vectorizer.fit_transform(data['stopword']).toarray()
        loaded_model = joblib.load('final_maxent_model.pkl')
        loaded_vectorizer = joblib.load('tfidf (1).pkl')


    
        with st.form("my_form"):
            st.subheader("Implementasi")
            # Judul Aplikasi
        st.title("Load CSV dengan Tombol")
        
        # Tambahkan tombol untuk load file CSV
        if st.button("Load CSV"):
            # Ganti "data.csv" dengan path file CSV Anda
            try:
                df = pd.read_csv("data.csv")
                st.success("CSV berhasil dimuat!")
                st.write(df)  # Tampilkan isi DataFrame
            except FileNotFoundError:
                st.error("File tidak ditemukan. Pastikan path file benar.")
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")


        
          


        
