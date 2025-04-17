# IMPORTS
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import time

# CONFIG PAGE
st.set_page_config(
    page_title="Classifieur de Prunes - Hackathon CDAI 2025",
    page_icon="🍑",
    layout="wide"
)

# CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #6B46C1;
        font-weight: 700;
        margin-bottom: 1rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4A5568;
        margin-bottom: 2rem;
        text-align: center;
    }
    .card, .result-card {
        background-color: #F7FAFC;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .confidence-high { color: #48BB78; font-weight: bold; }
    .confidence-medium { color: #ECC94B; font-weight: bold; }
    .confidence-low { color: #F56565; font-weight: bold; }
    .footer {
        text-align: center;
        margin-top: 30px;
        color: #718096;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# TITRES
st.markdown('<h1 class="main-header">🍑 TriPlumCamer</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Téléchargez l\'image d\'une prune pour analyser sa qualité</p>', unsafe_allow_html=True)

# CATEGORIES
categories = ["Non affectée", "Non mûre", "Tachetée", "Fissurée", "Meurtrie", "Pourrie"]
category_descriptions = {
    "Non affectée": "Prune mûre et saine, prête à être consommée ou vendue.",
    "Non mûre": "Prune pas encore mûre, nécessite plus de temps pour mûrir.",
    "Tachetée": "Prune avec des taches superficielles qui n'affectent pas son goût.",
    "Fissurée": "Prune avec des fissures qui peuvent affecter sa durée de conservation.",
    "Meurtrie": "Prune avec des meurtrissures ou contusions, qualité réduite.",
    "Pourrie": "Prune en état de décomposition, impropre à la consommation."
}
category_colors = {
    "Non affectée": "#48BB78",
    "Non mûre": "#38B2AC",
    "Tachetée": "#ECC94B",
    "Fissurée": "#ED8936",
    "Meurtrie": "#F56565",
    "Pourrie": "#9B2C2C"
}

# CHARGER LE MODELE
@st.cache_resource
def load_classification_model():
    try:
        # Corriger bug "groups=1" dans DepthwiseConv2D
        class CustomDepthwiseConv2D(DepthwiseConv2D):
            def __init__(self, *args, **kwargs):
                kwargs.pop('groups', None)
                super().__init__(*args, **kwargs)

        tf.keras.utils.get_custom_objects()['DepthwiseConv2D'] = CustomDepthwiseConv2D

        model = load_model("keras_model.h5")
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

# PRETRAITEMENT DE L'IMAGE
def preprocess_image(image, img_size=224):
    if isinstance(image, Image.Image):
        image = np.array(image)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    image = cv2.resize(image, (img_size, img_size))
    image = image / 255.0
    return np.expand_dims(image, axis=0)

# AFFICHAGE RESULTATS
def display_prediction_results(prediction, image):
    predicted_idx = np.argmax(prediction)
    predicted_class = categories[predicted_idx]
    confidence = float(prediction[0][predicted_idx])
    confidence_class = "confidence-high" if confidence > 0.7 else "confidence-medium" if confidence > 0.4 else "confidence-low"

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(image, caption="Image analysée", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f"### Résultat de l'analyse")
        st.markdown(f"**Classification :** {predicted_class}")
        st.markdown(f"**Confiance :** <span class='{confidence_class}'>{confidence:.2%}</span>", unsafe_allow_html=True)
        st.markdown(f"**Description :** {category_descriptions[predicted_class]}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Graphique
    st.markdown("### Probabilités pour chaque catégorie")
    fig, ax = plt.subplots(figsize=(10, 5))
    indices = np.argsort(prediction[0])[::-1]
    sorted_classes = [categories[i] for i in indices]
    sorted_probs = [prediction[0][i] for i in indices]
    sorted_colors = [category_colors[cat] for cat in sorted_classes]
    bars = ax.barh(sorted_classes, sorted_probs, color=sorted_colors)
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(min(width + 0.01, 1.0), bar.get_y() + bar.get_height()/2, f"{width:.2%}", va='center')
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Probabilité")
    ax.set_title("Distribution des probabilités")
    plt.tight_layout()
    st.pyplot(fig)

# APP PRINCIPALE
def main():
    model = load_classification_model()
    if model is None:
        return

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## Téléchargez une image de prune")
    upload_option = st.radio("Méthode de chargement :", ("Télécharger un fichier", "Utiliser la caméra"))
    image_data = None

    if upload_option == "Télécharger un fichier":
        uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image_data = Image.open(uploaded_file)
    else:
        camera_image = st.camera_input("Prenez une photo")
        if camera_image:
            image_data = Image.open(camera_image)
    st.markdown('</div>', unsafe_allow_html=True)

    if image_data:
        if st.button("Analyser la prune"):
            with st.spinner("Analyse en cours..."):
                time.sleep(1)
                processed = preprocess_image(image_data)
                prediction = model.predict(processed)
                display_prediction_results(prediction, image_data)
                st.success("Analyse terminée !")

    with st.expander("À propos de ce modèle"):
        st.markdown("""
        Ce modèle de deep learning a été entraîné sur des images de prunes africaines dans le cadre du Hackathon CDAI 2025.
        Il permet de détecter différentes classes de qualité :
        - **Non affectée**
        - **Non mûre**
        - **Tachetée**
        - **Fissurée**
        - **Meurtrie**
        - **Pourrie**
        """)

    st.markdown('<div class="footer">Développé pour le Hackathon CDAI 2025 | "Innover pour transformer"</div>', unsafe_allow_html=True)

# EXECUTION
if __name__ == "__main__":
    main()
