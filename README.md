🍑 Triplum Camer - Classifieur de Prunes Africaines
Bienvenue dans le projet Triplum Camer, une application développée pour le Hackathon CDAI 2025. Elle permet de classifier la qualité de prunes africaines à partir d’une simple image grâce à l’intelligence artificielle.

🎯 Objectif
Classer automatiquement une prune dans l'une des catégories suivantes :

✅ Non affectée

🟡 Non mûre

🔵 Tachetée

🟠 Fissurée

🔴 Meurtrie

🧠 Données et entraînement
Le modèle a été entraîné à l'aide du dataset de prunes africaines disponible sur Kaggle, intitulé : "Ensemble de données sur les prunes africaines". Ce jeu de données contient des images annotées représentant les différentes conditions possibles des prunes récoltées.

🚀 Pour exécuter le projet localement


git clone https://github.com/ChetchuengKamsuAI/triplumcamer.git
cd prune_class

# Création et activation de l'environnement virtuel
python -m venv venv
# Pour Linux / macOS
source venv/bin/activate
# Pour Windows
venv\Scripts\activate

# Installation des dépendances
pip install -r requirements.txt

# Lancement de l'application
streamlit run app.py
