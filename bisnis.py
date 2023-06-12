from streamlit_option_menu import option_menu
import streamlit as st
import pandas as pd 
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
<center><h2 style = "text-align: justify;">KLASIFIKASI KUALITAS GARAM PT.GARAM SUMENEP MENGGUNAKAN METODE DECISION TREE,NAIVE BAYES,K-NN,SVM</h2></center>
""",unsafe_allow_html=True)
st.write("### Dosen Pengampu : Eka Mala Sari Rochman, S.Kom., M.Kom.",unsafe_allow_html=True)

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
    if selected == "Implementation":

        
        # Menggunakan fungsi `read_csv` untuk memuat dataset dari file CSV
        data = pd.read_csv('https://raw.githubusercontent.com/Feb11F/dataset/main/data%20garam%20(1).csv')
        #Getting input from user
        st.write(data.head())
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


        #knn
        from sklearn.neighbors import KNeighborsClassifier

        # Membangun model KNN
        knn = KNeighborsClassifier(n_neighbors=5)

        # Melatih model dengan data latih
        knn.fit(X_train, y_train)

        # Memprediksi kelas target menggunakan data uji
        y_pred2 = knn.predict(X_test)

        # Menghitung akurasi prediksi
        accuracyknn = metrics.accuracy_score(y_test, y_pred2)

        # Menghitung akurasi prediksi
        accuracy = metrics.accuracy_score(y_test, y_pred)
        print("Akurasi: {:.2f}%".format(accuracy*100))

        #nb
        from sklearn.naive_bayes import GaussianNB
        

        # Membangun model Naive Bayes
        nb = GaussianNB()

        # Melatih model dengan data latih
        nb.fit(X_train, y_train)

        # Memprediksi kelas target menggunakan data uji
        y_pred3 = nb.predict(X_test)
        accuracynb = metrics.accuracy_score(y_test, y_pred3)
        #svm
        from sklearn.svm import SVC
        # Membangun model Naive Bayes
        ksvm = SVC(kernel = 'rbf')

        # Melatih model dengan data latih
        ksvm.fit(X_train, y_train)
        # Memprediksi kelas target menggunakan data uji
        y_pred_ksvm = ksvm.predict(X_test)
        accuracysvm = metrics.accuracy_score(y_test, y_pred_ksvm)
        

        print("Akurasi: {:.2f}%".format(accuracynb*100))
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
            inputs = np.array([kadar_air,tak_larut,kalsium,magnesium,sulfat,naclwb,nacldb])
            input_norm = np.array(inputs).reshape(1, -1)
            input_pred = clf.predict(input_norm)
            input_pred2 = knn.predict(input_norm)
            input_pred3 = nb.predict(input_norm)
            input_pred4 = ksvm.predict(input_norm)
            if submit:
                st.subheader('Hasil Prediksi')
            # Menampilkan hasil prediksi
                st.write('Decision Tree')
                st.success(input_pred[0])
                st.write("Akurasi: {:.2f}%".format(accuracy*100))

                st.write('KNN')
                st.success(input_pred2[0])
                st.write("Akurasi: {:.2f}%".format(accuracyknn*100))

                st.write('Naive Bayes')
                st.success(input_pred3[0])
                st.write("Akurasi: {:.2f}%".format(accuracynb*100))

                st.write('Support vector machine')
                st.success(input_pred4[0])
                st.write("Akurasi: {:.2f}%".format(accuracysvm*100))

                # Plot bar chart
                classifiers = ['Decision Tree', 'KNN', 'Naive Bayes', 'SVM']
                accuracies = [accuracy, accuracyknn, accuracynb, accuracysvm]

                fig, ax = plt.subplots()
                ax.bar(classifiers, accuracies)
                ax.set_ylabel('akurasi model')
                ax.set_title('perbandingan akurasi model')
                plt.ylim([0, 1])

                # Display plot using Streamlit
                st.pyplot(fig)
        
          


        
