# 🍑 Triplum Camer - Classifieur de Prunes Africaines

Bienvenue dans le projet **Triplum Camer**, une application développée pour le **Hackathon CDAI 2025**. Elle permet de classifier la qualité de prunes africaines à partir d’une simple image grâce à l’intelligence artificielle.

## 🎯 Objectif

Classer automatiquement une prune dans l'une des catégories suivantes :
- ✅ Non affectée
- 🟡 Non mûre
- 🔵 Tachetée
- 🟠 Fissurée
- 🔴 Meurtrie
- ⚫ Pourrie

## 🚀 Lancer l'application

### 1. Cloner le dépôt

```bash
git clone https://github.com/ChetchuengKamsuAI/triplumcamer.git
cd prune_class


python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows


pip install -r requirements.txt

streamlit run app.py
