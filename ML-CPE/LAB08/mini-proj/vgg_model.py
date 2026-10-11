import os
import json
from tensorflow.keras import layers, models

VGG16 = "VGG16"
VGG_SMALL = "VGG_SMALL"

def build_model(input_shape, num_classes, blocks_type):
    model = models.Sequential()
    
    # VGG-style Architecture (Simplified for speed on CPU/entry GPU)
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    
    # Output layer
    if num_classes <= 2:
        model.add(layers.Dense(1, activation='sigmoid'))
    else:
        model.add(layers.Dense(num_classes, activation='softmax'))
        
    return model

def train_model(X_train, y_train, X_val, y_val, num_classes, output_dir, epochs, batch_size, blocks):
    input_shape = X_train.shape[1:]
    model = build_model(input_shape, num_classes, blocks)
    
    loss_function = 'binary_crossentropy' if num_classes <= 2 else 'sparse_categorical_crossentropy'
    model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val)
    )
    
    # Save Model
    model.save(os.path.join(output_dir, "vgg_model.keras"))
    
    # Save History
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(hist_dict, f)
        
    return model, history

def predict_model(model, X_test):
    import numpy as np
    probabilities = model.predict(X_test)
    if probabilities.shape[-1] == 1:
        return (probabilities > 0.5).astype(int).ravel()
    else:
        return np.argmax(probabilities, axis=1)