from streamlit_option_menu import option_menu
import joblib
import streamlit as st
import pandas as pd 
from nltk.classify import MaxentClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
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
        st.write("#### Deskripsi Dataset")
        st.write(""" <p style = "text-align: justify;">Dataset yang digunakan merupakan data garam PT. Sumenep yang memiliki 7 fitur yaitu kadar air,tak larut,kalsium,magnesium,sulfat,NaCl(wb),NaCl(db) dan satu label yaitu grade</p>""",unsafe_allow_html=True)
        st.write("#### Kadar air")
        st.write(""" <p style = "text-align: justify;">Kadar air adalah persentase atau fraksi massa air yang terkandung dalam jumlah tertentu garam. Dalam konteks dataset garam, fitur kadar air biasanya digunakan untuk memberikan informasi tentang tingkat kelembaban garam tersebut.</p>""",unsafe_allow_html=True)
        st.write("#### Tak Larut")
        st.write(""" <p style = "text-align: justify;">Fitur tak larut pada dataset garam mengacu pada sifat-sifat fisikokimia yang tidak larut dalam air atau memiliki kelarutan yang sangat rendah dalam kondisi tertentu. Dalam konteks garam, fitur tak larut adalah kandungan zat-zat yang tetap padat atau tidak terlarut dalam larutan garam.</p>""",unsafe_allow_html=True)
        st.write("#### Kalsium")
        st.write(""" <p style = "text-align: justify;">Dalam konteks dataset garam, fitur kalsium mengacu pada konsentrasi atau jumlah kalsium yang terdeteksi dalam setiap sampel garam yang diukur atau dianalisis.</p>""",unsafe_allow_html=True)
        st.write("#### Magnesium")
        st.write(""" <p style = "text-align: justify;">Fitur magnesium dalam dataset garam merujuk pada kandungan atau nilai magnesium yang terdapat dalam garam tersebut. Magnesium adalah unsur kimia yang umumnya hadir dalam bentuk ion Mg2+ dalam senyawa garam.</p>""",unsafe_allow_html=True)   
        st.write("#### Sulfat")
        st.write(""" <p style = "text-align: justify;">Sulfat, yang secara kimia direpresentasikan dengan rumus SO4^2-, adalah garam asam sulfat (H2SO4) yang telah kehilangan dua proton (H+). Fitur ini mengacu pada konsentrasi atau jumlah sulfat yang terkandung dalam garam.</p>""",unsafe_allow_html=True)
        st.write("#### NaCl(wb)")
        st.write(""" <p style = "text-align: justify;"> fitur "NaCl(wb)" biasanya menunjukkan persentase kandungan natrium klorida dalam garam berbasis basah atau berat total garam termasuk air. Misalnya, jika terdapat data garam dengan nilai "NaCl(wb)" sebesar 98%, itu berarti bahwa 98% dari berat total garam tersebut adalah natrium klorida dan sisanya adalah air.</p>""",unsafe_allow_html=True)
        st.write("#### NaCl(db)")
        st.write(""" <p style = "text-align: justify;">"NaCl(db)" pada dataset garam mungkin menunjukkan bahwa tingkat kandungan NaCl diukur menggunakan skala desibel.</p>""",unsafe_allow_html=True) 
        data = pd.read_csv('https://raw.githubusercontent.com/Feb11F/dataset/main/data%20garam%20(1).csv')
        #Getting input from user
        st.write(data.head())
    if selected == "Implementation":
        import joblib

        loaded_model = joblib.load('final_maxent_model.pkl')
        loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')



        with st.form("my_form"):
            st.subheader("Implementasi")
            ulasan = st.text('Masukkan ulasan')
            new_X = loaded_vectorizer.transform(ulasan).toarray()

            # Mengubah fitur menjadi format yang sesuai dengan model SVM (menggunakan dictionary seperti yang diinginkan)
            # Membuat dictionary dengan nama feature yang mengikuti format yang diberikan
            new_data_features = {f"feature_{j}": new_X[0][j] for j in range(new_X.shape[1])}
            if submit:
                st.subheader('Hasil Prediksi')
            # Menampilkan hasil prediksi
                st.write('Decision Tree')
                st.success(input_pred[0])
                st.write("Akurasi: {:.2f}%".format(accuracy*100))


        
          


        
