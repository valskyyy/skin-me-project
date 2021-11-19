from  tensorflow.keras.models import load_model
model = load_model('skin_model.hdf5')

#-------

import streamlit as st
st.title('Bien dans ma peau')

st.subheader('BDMP permet aux particuliers de diagnostiquer de manière autonome leurs lésions de peau (type grain de beauté), d’évaluer un risque potentiel et de prendre rendez-vous chez un dermatologue si besoin.')

genre = st.radio(
"Vous êtes",
('Un homme', 'Une femme'))

age = st.slider('Votre âge', 0, 100, 1)

genre = st.radio(
"Où est localisée la lésion ?",
('Cuir chevelu', 'Oreille', 'Visage', 'Dos', 'Tronc', 'Poitrine', 'Membre supérieur', 'Abdomen', 'Membre inférieur', 'Zone génitale', 'Cou', 'Main', 'Pied', 'Zone acral'))


file = st.file_uploader("Uploader la photo (un seul grain de beauté par photo) :", type=["jpg", "png","jpeg"])

#----------

import cv2
from PIL import Image, ImageOps
import numpy as np
def import_and_predict(image_data, model):
    
        size = (128,128)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(128, 128),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("akiec : Actinic keratoses and intraepithelial carcinoma / Bowen's disease")
    elif np.argmax(prediction) == 1:
        st.write("bcc : basal cell carcinoma")
    elif np.argmax(prediction) == 2:
        st.write("bkl : benign keratosis-like lesions")
    elif np.argmax(prediction) == 3:
        st.write("df : dermatofibroma")
    elif np.argmax(prediction) == 4:
        st.write("mel : melanoma")
    elif np.argmax(prediction) == 5:
        st.write("nv : melanocytic nevi")
    else:
        st.write("vasc : vascular lesions")
    
    