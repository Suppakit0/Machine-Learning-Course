import os
import json
from tensorflow import keras
from tensorflow.keras import layers

def build_model(input_shape, num_classes):
    """สร้างโมเดล Deep Neural Network (DNN) สำหรับ Tabular Data"""
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(32, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Output layer สำหรับการจัดหมวดหมู่หลายคลาส (Multiclass)
        layers.Dense(num_classes, activation="softmax")
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train_model(X_train, y_train, X_val, y_val, num_classes, output_dir=None, epochs=50, batch_size=16):
    """ทำการเทรนโมเดลและบันทึกผลลัพธ์"""
    model = build_model((X_train.shape[1],), num_classes)
    model.summary()
    
    callbacks = [
        # หยุดการเทรนหาก val_loss ไม่ลดลง 10 epoch ติดต่อกัน
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ]
    
    print("\nTraining...")
    history = model.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val),
        epochs=epochs, 
        batch_size=batch_size, 
        callbacks=callbacks, 
        verbose=1
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        # บันทึก Model
        model.save(os.path.join(output_dir, "dnn_model.keras"))
        # บันทึก History
        with open(os.path.join(output_dir, "history.json"), "w") as f:
            json.dump({k: [float(v) for v in vs] for k, vs in history.history.items()}, f)
            
        print(f"\nSaved model to: {output_dir}/dnn_model.keras")
    
    return model, history

def predict_model(model, X_test):
    """ทำนายผลลัพธ์จากข้อมูลทดสอบ"""
    probabilities = model.predict(X_test, verbose=0)
    return probabilities.argmax(axis=1)