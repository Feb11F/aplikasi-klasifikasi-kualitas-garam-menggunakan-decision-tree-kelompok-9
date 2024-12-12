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
    page_icon='https://lh5.googleusercontent.com/p/AF1QipNgmGyncJl5jkHg6gnxdTSTqOKtIBpy-kl9PgDz=w540-h312-n-k-no',
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
        st.write("""<h3 style = "text-align: center;"></h3>""",unsafe_allow_html=True), 
        ["Home", "Dataset","prediksi ulasan","Implementation"], 
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
        <img src="https://lh5.googleusercontent.com/p/AF1QipNgmGyncJl5jkHg6gnxdTSTqOKtIBpy-kl9PgDz=w540-h312-n-k-no" width="500" height="300">
        </h3>""", unsafe_allow_html=True)
    if selected == "Dataset":
        file_path = 'data stopword tes.csv'  # Ganti dengan path ke file Anda
        data = pd.read_csv(file_path)
        st.write(data.head(10))
    if selected == "prediksi ulasan":
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
            ulasan = st.text_input('Masukkan ulasan')  # Input ulasan dari pengguna
            submit = st.form_submit_button("Prediksi")
            if submit:
                if ulasan.strip():  # Validasi input tidak kosong
                    # Transformasikan ulasan ke bentuk vektor
                    new_X = vectorizer.transform([ulasan]).toarray()
        
                    # Membuat dictionary dengan nama feature sesuai format model
                    new_data_features = {f"feature_{j}": new_X[0][j] for j in range(new_X.shape[1])}
                    
                    # Prediksi menggunakan model
                    new_pred = loaded_model.classify(new_data_features)
        
                    # Tampilkan hasil prediksi
                    st.subheader('Hasil Prediksi')
                    st.write(f"Prediction for New Data: {new_pred}")
                else:
                    st.error("Masukkan ulasan terlebih dahulu!")

    if selected == "Implementation":
        import joblib
        # Menggunakan pandas untuk membaca file CSV
        file_path = 'data stopword tes.csv'  # Ganti dengan path ke file Anda
        data = pd.read_csv(file_path)
        vectorizer = TfidfVectorizer(max_features=100)
        X = vectorizer.fit_transform(data['stopword']).toarray()
        loaded_model = joblib.load('final_maxent_model.pkl')
        loaded_vectorizer = joblib.load('tfidf (1).pkl')


    
        st.subheader("Implementasi")
            # Judul Aplikasi
        st.title("pilih sentimen wisata")
        
        # Tambahkan tombol untuk load file CSV
        if st.button("Bukit Jaddih"):
            try:
                df = pd.read_csv("bukit jaddih.csv")
            
                    # Mengambil 10 data pertama dari kolom 'ulasan'
                top_10_reviews = df['Ulasan'].head(10)
                
                # Menampilkan ulasan
                st.subheader("10 Ulasan Pertama")
                
                # Transformasi data ulasan ke fitur
                new_X = vectorizer.transform(top_10_reviews).toarray()
                
                # Membuat dictionary fitur (jika model membutuhkan format dictionary)
                features_list = [
                    {f"feature_{j}": new_X[i][j] for j in range(new_X.shape[1])} 
                    for i in range(new_X.shape[0])
                ]
                
                # Prediksi sentimen untuk setiap ulasan
                predictions = [loaded_model.classify(features) for features in features_list]
                
                # Menampilkan hasil prediksi
                st.subheader("Hasil Prediksi Sentimen")
                hasil_prediksi = pd.DataFrame({
                    "Ulasan": top_10_reviews,
                    "Prediksi Sentimen": predictions
                })
                st.write(hasil_prediksi)
                # Hitung mayoritas kelas
                from collections import Counter
                class_counts = Counter(predictions)
                majority_class = class_counts.most_common(1)[0][0]  # Kelas dengan frekuensi tertinggi

                st.write("Kelas mayoritas (kesimpulan):", majority_class)
            except FileNotFoundError:
                st.error("File tidak ditemukan. Pastikan path file benar.")
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")

        if st.button("Pantai Slopeng"):
            try:
                df = pd.read_csv("pantai slopeng.csv")
            
                    # Mengambil 10 data pertama dari kolom 'ulasan'
                top_10_reviews = df['Ulasan'].head(10)
                
                # Menampilkan ulasan
                st.subheader("10 Ulasan Pertama")
                
                # Transformasi data ulasan ke fitur
                new_X = vectorizer.transform(top_10_reviews).toarray()
                
                # Membuat dictionary fitur (jika model membutuhkan format dictionary)
                features_list = [
                    {f"feature_{j}": new_X[i][j] for j in range(new_X.shape[1])} 
                    for i in range(new_X.shape[0])
                ]
                
                # Prediksi sentimen untuk setiap ulasan
                predictions = [loaded_model.classify(features) for features in features_list]
                
                # Menampilkan hasil prediksi
                st.subheader("Hasil Prediksi Sentimen")
                hasil_prediksi = pd.DataFrame({
                    "Ulasan": top_10_reviews,
                    "Prediksi Sentimen": predictions
                })
                st.write(hasil_prediksi)
                # Hitung mayoritas kelas
                from collections import Counter
                class_counts = Counter(predictions)
                majority_class = class_counts.most_common(1)[0][0]  # Kelas dengan frekuensi tertinggi

                st.write("Kelas mayoritas (kesimpulan):", majority_class)
            except FileNotFoundError:
                st.error("File tidak ditemukan. Pastikan path file benar.")
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")

        if st.button("Api Tak Kunjung Padam"):
            try:
                df = pd.read_csv("api tak kunjung padam.csv")
            
                    # Mengambil 10 data pertama dari kolom 'ulasan'
                top_10_reviews = df['Ulasan'].head(10)
                
                # Menampilkan ulasan
                st.subheader("10 Ulasan Pertama")
                
                # Transformasi data ulasan ke fitur
                new_X = vectorizer.transform(top_10_reviews).toarray()
                
                # Membuat dictionary fitur (jika model membutuhkan format dictionary)
                features_list = [
                    {f"feature_{j}": new_X[i][j] for j in range(new_X.shape[1])} 
                    for i in range(new_X.shape[0])
                ]
                
                # Prediksi sentimen untuk setiap ulasan
                predictions = [loaded_model.classify(features) for features in features_list]
                
                # Menampilkan hasil prediksi
                st.subheader("Hasil Prediksi Sentimen")
                hasil_prediksi = pd.DataFrame({
                    "Ulasan": top_10_reviews,
                    "Prediksi Sentimen": predictions
                })
                st.write(hasil_prediksi)
                # Hitung mayoritas kelas
                from collections import Counter
                class_counts = Counter(predictions)
                majority_class = class_counts.most_common(1)[0][0]  # Kelas dengan frekuensi tertinggi

                st.write("Kelas mayoritas (kesimpulan):", majority_class)
            except FileNotFoundError:
                st.error("File tidak ditemukan. Pastikan path file benar.")
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")

        if st.button("Pantai Sembilan"):
            try:
                df = pd.read_csv("pantai sembilan.csv")
            
                    # Mengambil 10 data pertama dari kolom 'ulasan'
                top_10_reviews = df['Ulasan'].head(10)
                
                # Menampilkan ulasan
                st.subheader("10 Ulasan Pertama")
                
                # Transformasi data ulasan ke fitur
                new_X = vectorizer.transform(top_10_reviews).toarray()
                
                # Membuat dictionary fitur (jika model membutuhkan format dictionary)
                features_list = [
                    {f"feature_{j}": new_X[i][j] for j in range(new_X.shape[1])} 
                    for i in range(new_X.shape[0])
                ]
                
                # Prediksi sentimen untuk setiap ulasan
                predictions = [loaded_model.classify(features) for features in features_list]
                
                # Menampilkan hasil prediksi
                st.subheader("Hasil Prediksi Sentimen")
                hasil_prediksi = pd.DataFrame({
                    "Ulasan": top_10_reviews,
                    "Prediksi Sentimen": predictions
                })
                st.write(hasil_prediksi)
                # Hitung mayoritas kelas
                from collections import Counter
                class_counts = Counter(predictions)
                majority_class = class_counts.most_common(1)[0][0]  # Kelas dengan frekuensi tertinggi

                st.write("Kelas mayoritas (kesimpulan):", majority_class)
            except FileNotFoundError:
                st.error("File tidak ditemukan. Pastikan path file benar.")
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")

        if st.button("Air Terjun Toroan"):
            try:
                df = pd.read_csv("air terjun toroan.csv")
            
                    # Mengambil 10 data pertama dari kolom 'ulasan'
                top_10_reviews = df['Ulasan'].head(10)
                
                # Menampilkan ulasan
                st.subheader("10 Ulasan Pertama")
                
                # Transformasi data ulasan ke fitur
                new_X = vectorizer.transform(top_10_reviews).toarray()
                
                # Membuat dictionary fitur (jika model membutuhkan format dictionary)
                features_list = [
                    {f"feature_{j}": new_X[i][j] for j in range(new_X.shape[1])} 
                    for i in range(new_X.shape[0])
                ]
                
                # Prediksi sentimen untuk setiap ulasan
                predictions = [loaded_model.classify(features) for features in features_list]
                
                # Menampilkan hasil prediksi
                st.subheader("Hasil Prediksi Sentimen")
                hasil_prediksi = pd.DataFrame({
                    "Ulasan": top_10_reviews,
                    "Prediksi Sentimen": predictions
                })
                st.write(hasil_prediksi)
                # Hitung mayoritas kelas
                from collections import Counter
                class_counts = Counter(predictions)
                majority_class = class_counts.most_common(1)[0][0]  # Kelas dengan frekuensi tertinggi

                st.write("Kelas mayoritas (kesimpulan):", majority_class)
            except FileNotFoundError:
                st.error("File tidak ditemukan. Pastikan path file benar.")
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")

        if st.button("Pantai Lombang "):
            # Ganti "data.csv" dengan path file CSV Anda
            try:
                df = pd.read_csv("pantai lombang.csv")
            
                    # Mengambil 10 data pertama dari kolom 'ulasan'
                top_10_reviews = df['Ulasan'].head(10)
                
                # Menampilkan ulasan
                st.subheader("10 Ulasan Pertama")
                
                # Transformasi data ulasan ke fitur
                new_X = vectorizer.transform(top_10_reviews).toarray()
                
                # Membuat dictionary fitur (jika model membutuhkan format dictionary)
                features_list = [
                    {f"feature_{j}": new_X[i][j] for j in range(new_X.shape[1])} 
                    for i in range(new_X.shape[0])
                ]
                
                # Prediksi sentimen untuk setiap ulasan
                predictions = [loaded_model.classify(features) for features in features_list]
                
                # Menampilkan hasil prediksi
                st.subheader("Hasil Prediksi Sentimen")
                hasil_prediksi = pd.DataFrame({
                    "Ulasan": top_10_reviews,
                    "Prediksi Sentimen": predictions
                })
                st.write(hasil_prediksi)
                # Hitung mayoritas kelas
                from collections import Counter
                class_counts = Counter(predictions)
                majority_class = class_counts.most_common(1)[0][0]  # Kelas dengan frekuensi tertinggi

                st.write("Kelas mayoritas (kesimpulan):", majority_class)
            except FileNotFoundError:
                st.error("File tidak ditemukan. Pastikan path file benar.")
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")


        
          


        
