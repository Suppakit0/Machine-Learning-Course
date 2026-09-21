import matplotlib
matplotlib.use("Agg") # เซ็ตเพื่อไม่ให้มีปัญหาตอนเซฟรูป
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(y_test, predictions, classes, save_path=None):
    """ประเมินผลความแม่นยำและสร้าง Confusion Matrix"""
    accuracy = accuracy_score(y_test, predictions)
    
    print("\n------------ Evaluation ------------------")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=classes, zero_division=0))
    
    matrix = confusion_matrix(y_test, predictions)
    
    if save_path:
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved Confusion Matrix: {save_path}")
        
    return accuracy

def plot_history(history, save_path):
    """วาดกราฟ Loss และ Accuracy ของการเทรน"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    
    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved Training History: {save_path}")