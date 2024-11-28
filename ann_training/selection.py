import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from data.training_data import lyrics, albums
from data.test_data import test_lyrics, test_albums

from data.custom_data import custom_lyrics, custom_albums

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000)
tokenizer.fit_on_texts(lyrics)

def preprocess_data(lyrics, albums, album_dict):
    sequences = tokenizer.texts_to_sequences(lyrics)
    X = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')
    y = [album_dict[album] for album in albums]
    y = tf.keras.utils.to_categorical(y, num_classes=len(album_dict))
    return X, y

# Diccionario dinámico
unique_albums = sorted(set(albums))
album_dict = {album: i for i, album in enumerate(unique_albums)}

# Preprocesar conjuntos de datos
X_train, y_train = preprocess_data(lyrics, albums, album_dict)
X_test, y_test = preprocess_data(test_lyrics, test_albums, album_dict)
X_custom, y_custom = preprocess_data(custom_lyrics, custom_albums, album_dict)

model_scores = []

for model_file in os.listdir("ann"):
    if model_file.endswith(".h5"):
        model_path = os.path.join("ann", model_file)
        model = load_model(model_path)

        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        custom_loss, custom_acc = model.evaluate(X_custom, y_custom, verbose=0)

        weighted_score = 0.5 * test_acc + 0.3 * custom_acc + 0.2 * train_acc
        model_scores.append((model_file, weighted_score, test_acc, custom_acc, train_acc))
        print(f"Modelo: {model_file} | Test Acc: {test_acc:.4f}, Custom Acc: {custom_acc:.4f}, Train Acc: {train_acc:.4f}, Weighted Score: {weighted_score:.4f}")

best_model = max(model_scores, key=lambda x: x[1])
print("\n=== Mejor Modelo ===")
print(f"Modelo: {best_model[0]}")
print(f"Test Acc: {best_model[2]:.4f}, Custom Acc: {best_model[3]:.4f}, Train Acc: {best_model[4]:.4f}, Weighted Score: {best_model[1]:.4f}")
