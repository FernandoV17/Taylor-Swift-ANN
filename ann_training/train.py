import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dropout, Bidirectional
from sklearn.model_selection import train_test_split
from data.training_data import lyrics, albums
from data.test_data import test_lyrics, test_albums

os.makedirs("ann", exist_ok=True)

np.random.seed(42)
tf.random.set_seed(42)

unique_albums = sorted(set(albums))
album_dict = {album: i for i, album in enumerate(unique_albums)}

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(lyrics)
sequences = tokenizer.texts_to_sequences(lyrics)
X = pad_sequences(sequences, padding='post')
y = [album_dict[album] for album in albums]
y = tf.keras.utils.to_categorical(y, num_classes=len(unique_albums))

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

test_sequences = tokenizer.texts_to_sequences(test_lyrics)
X_test = pad_sequences(test_sequences, padding='post')
y_test = [album_dict[album] for album in test_albums]
y_test = tf.keras.utils.to_categorical(y_test, num_classes=len(unique_albums))

def build_model():
    model = models.Sequential([
        layers.Embedding(input_dim=5000, output_dim=128, input_length=X.shape[1]),
        Bidirectional(layers.LSTM(128, return_sequences=False)),
        Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(len(unique_albums), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

best_accuracy = 0
threshold_accuracy = 0.85
generations = 5

for gen in range(generations):
    print(f"\n=== Generación {gen + 1} ===\n")
    model = build_model()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=16,
        verbose=1
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Precisión en datos de prueba: {accuracy:.4f}")

    model_path = f"ann/modelo_gen_{gen + 1}.h5"
    model.save(model_path)
    print(f"Modelo guardado en: {model_path}")

    if accuracy >= threshold_accuracy:
        print(f"Se alcanzó una precisión aceptable: {accuracy:.4f}")
        break
