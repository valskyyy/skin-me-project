from  tensorflow.keras.models import load_model
model = load_model('https://www.mydatapassion.com/model256_3classes_20epochs.h5')
import joblib
import tensorflow as tf
import pandas as pd
import itertools
enc = joblib.load('encoder.joblib')

#-------
#Dermatologist API URL
import requests
derm_api_url = "https://public.opendatasoft.com/api/records/1.0/search/?dataset=medecins&q=&facet=civilite&facet=column_12&facet=column_13&facet=column_14&facet=column_16&facet=libelle_profession&facet=type_dacte_realise&facet=commune&facet=nom_epci&facet=nom_dep&facet=nom_reg&facet=insee_reg&facet=insee_dep&facet=libelle_regroupement&facet=libelle&facet=libelle_acte_clinique&facet=l&refine.commune={}&refine.libelle_profession=Dermatologue+et+vénérologue"
#-------

#Template for dermatologist search results

DERMATOLOGIST_SEARCH_RESULTS = """
<div style = width:100%;height:100%;margin:1px;padding:5px;margin:10px;position:relative;border-radius:5px;background-color:#f0f2f6>
<h5>{}<br /></h5>
<h6>{}<br /></h6>
<h7>{}<br /></h7>
<h7>{}<br /></h7>
<h8>{}<br /></h8>
</div>
"""

import streamlit as st
st.title('Bienvenue sur SkinMe !')

st.subheader("SkinMe vous permet d’établir un premier diagnostic de vos lésions de la peau, d’évaluer un risque potentiel et de prendre rendez-vous chez un dermatologue si besoin.")

gender = st.radio(
"Vous êtes",
('Un homme', 'Une femme'))

age = st.number_input('Votre age', min_value = 0, max_value = 84, value = 30, step = 1)

localization = st.radio(
"Localisation de la lésion",
('Cuir chevelu', 'Oreille', 'Visage', 'Dos', 'Buste', 'Poitrine', 'Membre supérieur', 'Abdomen', 'Membre inférieur', 'Zone génitale', 'Cou', 'Main', 'Pied'))


file = st.file_uploader("Choisissez votre photo (assurez-vous d’avoir une seule lésion par photo) :", type=["jpg", "png","jpeg"])
st.write(gender, age, localization)
#----------

import cv2
from PIL import Image, ImageOps
import numpy as np

age_input = (age // 5) * 5
gender_input = ['male' if gender == 'homme' else 'female']
localization_input = ['back' if localization == 'Dos'
else 'lower extremity' if localization == 'Membre inférieur'
else 'trunk' if localization == 'Buste'
else 'upper extremity' if localization == 'Membre supérieur'
else 'abdomen' if localization == 'Abdomen'
else 'face' if localization == 'Visage'
else 'chest' if localization == 'Poitrine'
else 'foot' if localization == 'Pied'
else 'scalp' if localization == 'Cuir chevelu'
else 'neck' if localization == 'Cou'
else 'hand' if localization == 'Main'
else 'genital' if localization == 'Zone génitale'
else 'ear' if localization == 'Oreille'
else 'unknown']

user_inputs={'age': age_input,'sex': gender_input,'localization2': localization_input}
tab_inputs = pd.DataFrame(user_inputs, index=[0])

tab_enc_inputs = enc.transform(tab_inputs)

tabular = [tf.reshape(tf.convert_to_tensor(tab_enc_inputs, name="tensor_cat_features"),[1,-1])]

#Function to make the prediction

def import_and_predict(image_data, tabular_data, model):
    img_size = 256
    channels = 3
    len_X = 36
    image = ImageOps.fit(image_data, (img_size, img_size), Image.ANTIALIAS)

    image = np.expand_dims(image, axis=0)
    # Should confirm what the line below does (if it's necessary or not)
    # img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = tf.data.Dataset.from_tensor_slices(tf.reshape(image, (1, 1, img_size, img_size, channels)))
    # Don't forget to put the last two dimensions as variables at the beginning of the function
    tabular = tf.data.Dataset.from_tensor_slices(
        tf.reshape(tabular_data, (1, 1, 1, len_X))
    )

    input = tf.data.Dataset.zip({"input_image": image, "input_tabular": tabular})
    prediction = model.predict(input)

    return prediction

#Streamlit interface

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, tabular, model)
    
    if np.argmax(prediction) == 1:
        st.title('Votre diagnostic:')
        st.success("Cette lésion semble bénigne. N’hésitez toutefois pas à consulter un dermatologue à la moindre tache suspecte.")
        st.write("&#128073; Notre évaluation n’est en aucun cas un diagnostic médical. Elle doit être confirmée par un médecin. N'hésitez pas à consulter si vous avez le moindre doute.")
    elif np.argmax(prediction) == 0:
        st.title('Votre diagnostic:')
        st.error("Le diagnostic de votre lésion semble mauvais. Nous vous conseillons de repidement prendre rendez-vous chez un dermatologue afin de compléter le diagnostic par un professionnel.")
        st.write("&#128073; Notre évaluation n’est en aucun cas un diagnostic médical. Elle doit être confirmée par un médecin.")
        #Find a dermatologist form
        
        st.header('Trouver un dermatologue:')
        with st.form(key="searchform"):
            first, second = st.columns([2,1])
            with first:
                city = st.text_input('Où ça ?:', placeholder="ex : '75009', '59', 'Lille', 'Creuse'...")
            with second:
                st.text(".")
                submit = st.form_submit_button("Recherche")
        #Results of the search
        col1, col2 = st.columns([2,1])
        with col1:
            if submit:
                search_url = derm_api_url.format(city)
                request = requests.get("https://public.opendatasoft.com/api/records/1.0/search/?dataset=medecins&q={}&rows=100&facet=civilite&facet=column_12&facet=column_13&facet=column_14&facet=column_16&facet=libelle_profession&facet=type_dacte_realise&facet=commune&facet=nom_epci&facet=nom_dep&facet=nom_reg&facet=insee_reg&facet=insee_dep&facet=libelle_regroupement&facet=libelle&facet=libelle_acte_clinique&facet=dep&refine.libelle_profession=Dermatologue+et+vénérologue".format(city))
                data = request.json()
                number_results = len(data['records'])
                st.subheader("{} dermatologues trouvés:".format(number_results, city))

                for i in range(0,number_results):
                    name = data['records'][i]['fields']['nom']
                    #st.write(name)
                    job = data['records'][i]['fields']['libelle_profession']
                    #st.write(job)
                    adress = data['records'][i]['fields']['adresse']
                    #st.write(adress)
                    try:
                        tel = data['records'][i]['fields']['column_10']
                    except KeyError:
                        print (" ")
                    #st.write(tel)
                    conv = data['records'][i]['fields']['column_14']
                    st.markdown(DERMATOLOGIST_SEARCH_RESULTS.format(name,job,adress,tel,conv),unsafe_allow_html=True)
    

    elif np.argmax(prediction) == 2:
        st.title('Votre diagnostic:')
        st.warning(" La lésion que nous avons analysée est potentiellement dangereuse. Nous vous conseillons de demander un avis médical sans attendre.")
        st.write("&#128073; Notre évaluation n’est en aucun cas un diagnostic médical. Elle doit être confirmée par un médecin.")
        #Find a dermatologist form
        
        st.header('Trouver un dermatologue:')
        with st.form(key="searchform"):
            first, second = st.columns([2,1])
            with first:
                city = st.text_input('Où ça ?:', placeholder="ex : '75009', '59', 'Lille', 'Creuse'...")
            with second:
                st.text(".")
                submit = st.form_submit_button("Recherche")
        #Results of the search
        col1, col2 = st.columns([2,1])
        with col1:
            if submit:
                search_url = derm_api_url.format(city)
                request = requests.get("https://public.opendatasoft.com/api/records/1.0/search/?dataset=medecins&q={}&rows=100&facet=civilite&facet=column_12&facet=column_13&facet=column_14&facet=column_16&facet=libelle_profession&facet=type_dacte_realise&facet=commune&facet=nom_epci&facet=nom_dep&facet=nom_reg&facet=insee_reg&facet=insee_dep&facet=libelle_regroupement&facet=libelle&facet=libelle_acte_clinique&facet=dep&refine.libelle_profession=Dermatologue+et+vénérologue".format(city))
                data = request.json()
                number_results = len(data['records'])
                st.subheader("{} dermatologues trouvés:".format(number_results, city))

                for i in range(0,number_results):
                    name = data['records'][i]['fields']['nom']
                    #st.write(name)
                    job = data['records'][i]['fields']['libelle_profession']
                    #st.write(job)
                    adress = data['records'][i]['fields']['adresse']
                    #st.write(adress)
                    try:
                        tel = data['records'][i]['fields']['column_10']
                    except KeyError:
                        print (" ")
                    #st.write(tel)
                    conv = data['records'][i]['fields']['column_14']
                    st.markdown(DERMATOLOGIST_SEARCH_RESULTS.format(name,job,adress,tel,conv),unsafe_allow_html=True)
    