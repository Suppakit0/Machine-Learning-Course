# ML-10-Object-Detection-II

This project implements **YOLOv8-based object detection** using the Aquarium Dataset. It focuses on training, evaluating, and deploying a machine learning model to detect and classify various aquatic animals in images and real-time video streams.

## 1. Project Overview

The project fine-tunes the **Ultralytics YOLOv8n** pretrained model to detect 7 distinct classes of aquatic animals. It includes scripts for training the model, evaluating its performance on a test set, performing inference on static images, and running real-time detection via a webcam.

**Features:**
- Custom object detection for 7 aquatic classes.
- Pretrained YOLOv8 fine-tuning with GPU acceleration (CUDA).
- Performance evaluation using Precision, Recall, and mAP metrics.
- Inference scripts for both random test images and live webcam feed.
- Comparison of model performance across different training epochs (10 vs 30).

## 2. Dataset

The **[Aquarium Dataset](https://universe.roboflow.com/brad-dwyer/aquarium-combined/dataset/2)** (sourced via Roboflow) is used in YOLO annotation format.

**Classes (7):** 
`fish`, `jellyfish`, `penguin`, `puffin`, `shark`, `starfish`, `stingray`

**Data Split:**
- **Training:** 448 images
- **Validation:** 127 images
- **Test:** 63 images

## 3. Project Structure

```text
LAB10/
├── Dataset/
│   └── Aquarium/
│       ├── train/
│       ├── valid/
│       └── test/
└── m-project/
    ├── data.yaml
    ├── train.py
    ├── evaluate.py
    ├── image_test.py
    ├── realtime_test.py
    └── runs/               # Output directory for weights and results
