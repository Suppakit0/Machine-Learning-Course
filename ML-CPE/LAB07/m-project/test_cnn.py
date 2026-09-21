import json
import os
import numpy as np
from tensorflow import keras

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

def test_model(n_samples=5):
    """สุ่มดึงข้อมูล Test มาทดสอบกับโมเดลที่เทรนเสร็จแล้ว"""
    print("Loading saved model and test data...")
    
    # โหลด Model
    model_path = os.path.join(OUTPUT_DIR, "dnn_model.keras")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please run main.py first.")
        return
        
    model = keras.models.load_model(model_path)
    
    # โหลดข้อมูลทดสอบ (Test set) และชื่อคลาส
    X_test = np.load(f"{OUTPUT_DIR}/X_test.npy")
    y_test = np.load(f"{OUTPUT_DIR}/y_test.npy")
    
    with open(f"{OUTPUT_DIR}/classes.json") as f:
        classes = json.load(f)
        
    # สุ่มข้อมูลตามจำนวน n_samples
    n_samples = min(n_samples, len(X_test)) # เผื่อกรณีข้อมูล test มีน้อยกว่า 5
    index = np.random.choice(len(X_test), n_samples, replace=False)
    X_sample = X_test[index]
    y_sample = y_test[index]
    
    # ทำนายผล
    probabilities = model.predict(X_sample, verbose=0)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    
    print("\n--- Test Random Samples Results ---")
    correct_count = 0
    
    for i in range(n_samples):
        pred_label = classes[predictions[i]]
        true_label = classes[y_sample[i]]
        
        correct = (predictions[i] == y_sample[i])
        if correct:
            correct_count += 1
            
        status = "✅ OK" if correct else "❌ WRONG"
        
        print(f"Sample [{i+1}] | Predicted: {pred_label:<7} | True: {true_label:<7} | Confidence: {confidence[i]*100:5.1f}% | {status}")
        
    print(f"\nTotal Correct: {correct_count}/{n_samples} ({(correct_count/n_samples)*100:.1f}%)")

if __name__ == "__main__":
    test_model()