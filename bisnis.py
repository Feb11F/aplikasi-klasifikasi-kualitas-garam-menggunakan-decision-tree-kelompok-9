from streamlit_option_menu import option_menu
import streamlit as st
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import numpy as np
import warnings
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
<center><h2 style = "text-align: CENTER;">KLASIFIKASI KUALITAS GARAM MENGGUNAKAN DECISION TREE </h2></center>
""",unsafe_allow_html=True)
st.write("### Dosen Pengampu : Dr. FIKA HASTARITA RACHMAN, ST., M.Eng",unsafe_allow_html=True)

with st.container():
    with st.sidebar:
        selected = option_menu(
        st.write("""<h3 style = "text-align: center;"><img src="https://cdn-icons-png.flaticon.com/512/1998/1998664.png" width="120" height="120"></h3>""",unsafe_allow_html=True), 
        ["Implementation"], 
            icons=['house', 'bar-chart','check2-square', 'person'], menu_icon="cast", default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#412a7a"},
                "icon": {"color": "white", "font-size": "18px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "color":"white"},
                "nav-link-selected":{"background-color": "#412a7a"}
            }
        )


    if selected == "Implementation":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            inputs=[]
            input_norm=[]

            # Menggunakan fungsi `read_csv` untuk memuat dataset dari file CSV
            data = pd.read_csv('https://raw.githubusercontent.com/Feb11F/dataset/main/data%20garam%20(1).csv')
            #Getting input from user


            # Memuat dataset 
            X = data.drop(['Grade','Data'],axis=1)
            y = data['Grade'] # Kelas

            # Membagi dataset menjadi data latih dan data uji
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

            # Membuat objek Decision Tree Classifier
            clf = DecisionTreeClassifier(max_depth=9)

            # Melatih Decision Tree Classifier menggunakan data latih
            clf.fit(X_train, y_train)

            # Memprediksi kelas untuk data uji
            y_pred = clf.predict(X_test)

            # Menghitung akurasi prediksi
            accuracy = metrics.accuracy_score(y_test, y_pred)
            print("Akurasi: {:.2f}%".format(accuracy*100))
            with st.form("my_form"):
                st.subheader("Implementasi")
                kadar_air = st.number_input('Masukkan kadar air')
                tak_larut = st.number_input('Masukkan kandungan zat tak larut')
                kalsium = st.number_input('Masukkan kandungan kalsium')
                magnesium = st.number_input('Masukkan kandungan magnesium')
                sulfat = st.number_input('Masukkan kandungan sulfat')
                naclwb = st.number_input('Masukkan kandungan NaCl wb')
                nacldb = st.number_input('Masukkan kandungan NaCl db')
                submit = st.form_submit_button("submit")
                if submit:
                    inputs = np.array([kadar_air,tak_larut,kalsium,magnesium,sulfat,naclwb,nacldb])
                    input_norm = np.array(inputs).reshape(1, -1)

                input_pred = clf.predict(input_norm)


                st.subheader('Hasil Prediksi')
                # Menampilkan hasil prediksi
                st.write("Data uji: Kualitas air", input_pred[0])
                st.write("Akurasi: {:.2f}%".format(accuracy*100))






