# Exploration des données et modèle de base pour la classification des prunes
# Hackathon CDAI 2025

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import random

# Configuration des chemins

BASE_DIR = "C:\\Users\\TOUTENUN\\Downloads\\prune\\african_plums_dataset" # Modifiez avec le chemin de votre dataset
categories = ["unaffected", "unripe", "spotted", "cracked", "bruised", "rotten"]

# Partie 1: Exploration des données
# =====================================

def load_dataset_info():
    """Analyse la structure du dataset et compte les images par catégorie"""
    image_counts = {}
    image_paths = []
    labels = []
    
    for idx, category in enumerate(categories):
        category_path = os.path.join(BASE_DIR, category)
        files = os.listdir(category_path)
        image_counts[category] = len(files)
        
        # Collecter les chemins d'images et les étiquettes
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(category_path, file))
                labels.append(idx)
    
    return image_counts, image_paths, labels

# Exécuter l'analyse du dataset
image_counts, image_paths, labels = load_dataset_info()

# Afficher la répartition des classes
plt.figure(figsize=(12, 6))
sns.barplot(x=list(image_counts.keys()), y=list(image_counts.values()))
plt.title('Nombre d\'images par catégorie de prunes')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Afficher quelques exemples d'images de chaque catégorie
plt.figure(figsize=(15, 10))
for i, category in enumerate(categories):
    category_path = os.path.join(BASE_DIR, category)
    files = [f for f in os.listdir(category_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Prendre 3 images aléatoires de chaque catégorie
    sample_files = random.sample(files, min(3, len(files)))
    
    for j, file in enumerate(sample_files):
        img_path = os.path.join(category_path, file)
        img = Image.open(img_path)
        plt.subplot(len(categories), 3, i*3 + j + 1)
        plt.imshow(img)
        plt.title(f"{category}")
        plt.axis('off')
plt.tight_layout()
plt.show()

# Analyser les dimensions des images
def analyze_image_dimensions():
    widths = []
    heights = []
    
    # Prendre un échantillon aléatoire pour l'analyse
    sample_paths = random.sample(image_paths, min(100, len(image_paths)))
    
    for path in sample_paths:
        img = Image.open(path)
        width, height = img.size
        widths.append(width)
        heights.append(height)
    
    return widths, heights

widths, heights = analyze_image_dimensions()

# Visualiser les dimensions
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(widths, bins=20)
plt.title('Distribution des largeurs d\'images')
plt.xlabel('Largeur (pixels)')
plt.ylabel('Nombre d\'images')

plt.subplot(1, 2, 2)
plt.hist(heights, bins=20)
plt.title('Distribution des hauteurs d\'images')
plt.xlabel('Hauteur (pixels)')
plt.ylabel('Nombre d\'images')
plt.tight_layout()
plt.show()

# Analyser les canaux de couleur (exemple sur quelques images)
def analyze_color_channels():
    # Sélectionner une image de chaque catégorie
    sample_images = []
    for category in categories:
        category_path = os.path.join(BASE_DIR, category)
        files = [f for f in os.listdir(category_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if files:
            sample_file = random.choice(files)
            img_path = os.path.join(category_path, sample_file)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir de BGR à RGB
            sample_images.append((category, img))
    
    # Afficher les histogrammes de couleur
    plt.figure(figsize=(15, 10))
    for i, (category, img) in enumerate(sample_images):
        plt.subplot(len(sample_images), 2, 2*i+1)
        plt.imshow(img)
        plt.title(f"{category}")
        plt.axis('off')
        
        plt.subplot(len(sample_images), 2, 2*i+2)
        color = ('r', 'g', 'b')
        for j, col in enumerate(color):
            hist = cv2.calcHist([img], [j], None, [256], [0, 256])
            plt.plot(hist, color=col)
            plt.xlim([0, 256])
        plt.title(f'Histogramme {category}')
    plt.tight_layout()
    plt.show()

analyze_color_channels()

# Partie 2: Préparation des données et modèle de base
# ===================================================

# Paramètres pour le modèle
IMG_SIZE = 320  # Taille standard pour de nombreux modèles CNN
BATCH_SIZE = 32
EPOCHS = 15
NUM_CLASSES = len(categories)

def prepare_data():
    # Préparer les chemins et les étiquettes
    X = np.array(image_paths)
    y = np.array(labels)
    
    # Diviser en ensembles d'entraînement, de validation et de test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()

print(f"Ensemble d'entraînement: {len(X_train)} images")
print(f"Ensemble de validation: {len(X_val)} images")
print(f"Ensemble de test: {len(X_test)} images")

# Générateurs d'images pour l'entraînement avec augmentation de données
def create_generators():
    # Générateur pour l'entraînement avec augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Générateurs pour validation et test (seulement rescale)
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Fonction pour charger et prétraiter une image
    def load_image(path, target_size=(IMG_SIZE, IMG_SIZE)):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        return img
    
    # Générateurs de flux
    def generate_batches(X, y, batch_size, datagen, is_train=False):
        num_samples = len(X)
        indices = np.arange(num_samples)
        if is_train:
            np.random.shuffle(indices)
        
        while True:
            for start_idx in range(0, num_samples, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                batch_X = np.array([load_image(X[i]) for i in batch_indices])
                batch_y = to_categorical(y[batch_indices], num_classes=NUM_CLASSES)
                
                if is_train:
                    # Appliquer l'augmentation de données
                    for x in datagen.flow(batch_X, batch_y, batch_size=len(batch_indices), shuffle=False):
                        batch_X, batch_y = x
                        break
                
                yield batch_X, batch_y
    
    train_generator = generate_batches(X_train, y_train, BATCH_SIZE, train_datagen, is_train=True)
    val_generator = generate_batches(X_val, y_val, BATCH_SIZE, val_test_datagen)
    test_generator = generate_batches(X_test, y_test, BATCH_SIZE, val_test_datagen)
    
    return train_generator, val_generator, test_generator

train_generator, val_generator, test_generator = create_generators()

# Création du modèle CNN de base
def create_model():
    model = Sequential([
        # Premier bloc de convolution
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Deuxième bloc de convolution
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Troisième bloc de convolution
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Couches entièrement connectées
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    # Compiler le modèle
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Créer et afficher le résumé du modèle
model = create_model()
model.summary()

# Callbacks pour l'entraînement
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=0.00001
)

# Entrainer le modèle
# Décommentez les lignes suivantes pour entrainer le modèle

"""
# Calcul des étapes par époque
steps_per_epoch = len(X_train) // BATCH_SIZE
validation_steps = len(X_val) // BATCH_SIZE

# Entraînement du modèle
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=validation_steps,
    callbacks=[early_stopping, reduce_lr]
)

# Visualiser les performances d'entraînement
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Précision du modèle')
plt.ylabel('Précision')
plt.xlabel('Époque')
plt.legend(['Entraînement', 'Validation'], loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Perte du modèle')
plt.ylabel('Perte')
plt.xlabel('Époque')
plt.legend(['Entraînement', 'Validation'], loc='upper right')
plt.tight_layout()
plt.show()

# Évaluation sur l'ensemble de test
test_steps = len(X_test) // BATCH_SIZE
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_steps)
print(f"Précision sur l'ensemble de test: {test_accuracy:.4f}")

# Sauvegarder le modèle
model.save('plum_classification_base_model.h5')
print("Modèle sauvegardé sous 'plum_classification_base_model.h5'")
"""

# Pour prédire sur une image spécifique (à titre d'exemple)
def predict_image(model, image_path):
    # Charger et prétraiter l'image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Prédire
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    class_name = categories[predicted_class]
    confidence = prediction[0][predicted_class]
    
    # Afficher l'image et la prédiction
    plt.figure(figsize=(6, 6))
    plt.imshow(img[0])
    plt.title(f"Prédiction: {class_name}\nConfiance: {confidence:.2f}")
    plt.axis('off')
    plt.show()
    
    return class_name, confidence

# Exemple d'utilisation (décommentez pour tester)
# image_to_predict = "chemin/vers/image/test.jpg"
# predict_image(model, image_to_predict)