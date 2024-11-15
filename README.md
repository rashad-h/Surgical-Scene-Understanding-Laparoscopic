
# Surgical Scene Understanding via Deep Learning

## Description

This repository features a U-Net-based segmentation model designed to identify surgical instruments and anatomical structures in laparoscopic hysterectomy procedures. The model uses the AutoLaparo dataset to train and evaluate its performance, leveraging state-of-the-art techniques to address challenges such as limited visualization, class imbalance, and overfitting.

### Dataset Reference
The dataset used is **AutoLaparo**:
```
@InProceedings{wang2022autolaparo,
    title = {AutoLaparo: A New Dataset of Integrated Multi-tasks for Image-guided Surgical Automation in Laparoscopic Hysterectomy},
    author = {Wang, Ziyi and Lu, Bo and Long, Yonghao and Zhong, Fangxun and Cheung, Tak-Hong and Dou, Qi and Liu, Yunhui},
    booktitle = {International Conference on Medical Image Computing and Computer-Assisted Intervention},
    pages = {486--496},
    year = {2022},
    organization = {Springer}
}
```

### Video Demo
Watch the model in action: [YouTube Video](https://www.youtube.com/watch?v=PnGtaq9bOVY)

## Features and Techniques

- **U-Net Architecture**: Implements a convolutional encoder-decoder network with skip connections to integrate spatial and abstract features for precise segmentation.
- **Custom Loss Functions**: Utilizes Weighted Cross-Entropy and Focal Loss to address class imbalance.
- **Real-Time Video Processing**: Applies segmentation to laparoscopic videos, generating RGB mask overlays for visualization.
- **Data Augmentation**: Enhances training diversity with transformations like flipping, rotation, and brightness adjustment.
- **Evaluation Metrics**: Includes IoU and Dice Coefficient for comprehensive model evaluation.

## Model Architecture

### Overview
The model architecture is based on U-Net, a convolutional neural network tailored for semantic segmentation tasks. Below is a detailed breakdown:

1. **Input Layer**:
   - Input images resized to 256×256×3 (RGB).
   - Normalized pixel values for smoother convergence.

2. **Encoder**:
   - Convolutional layers with 3×3 filters and ReLU activation functions.
   - Max-pooling layers to downsample spatial dimensions and increase feature depth.
   - Dropout layers to mitigate overfitting.

3. **Bottleneck**:
   - Captures compact feature representations at the lowest resolution (16×16×256).
   - Incorporates deeper convolutions and dropout for robustness.

4. **Decoder**:
   - Transposed convolutional layers to upsample feature maps.
   - Concatenates earlier encoder layers for combining low-level spatial and high-level abstract features.

5. **Output Layer**:
   - Produces segmentation masks matching the input resolution (256×256).
   - Uses a Softmax activation function to assign pixel-wise probabilities across 10 surgical classes.

### Diagram of the Architecture
![Model Architecture](https://github.com/rashad-h/Surgical-Scene-Understanding-Laparoscopic/assets/61196340/12e9a707-314b-4a4b-a2fd-e0a18306496d)

### Key Features
- **Skip Connections**: Improve localization by bridging encoder and decoder layers.
- **Softmax Activation**: Normalizes pixel probabilities across classes.
- **Adaptive Learning**: Utilizes learning rate adjustments to enhance training efficiency.

## Insights from the Report

- **Problem**: Laparoscopic surgeries rely heavily on video feeds, where limited visibility and depth perception make navigation challenging.
- **Results**:
  - Achieved 69% mean IoU and 72% Dice Coefficient on the test dataset.
  - Addressed class imbalance with weighted loss functions, improving minor category segmentation.
- **Challenges**:
  - Performance degradation in low-light and smoky conditions.
  - Inconsistent segmentation in real-time video due to jittering.
- **Future Work**:
  - Integrate Attention U-Net for better feature extraction.
  - Incorporate temporal consistency using RNNs for smoother video segmentation.
  - Expand dataset diversity to improve generalization.

## Features and Techniques

- **U-Net Architecture**: Implements a convolutional encoder-decoder network with skip connections for accurate segmentation.
- **Custom Loss Functions**: Utilizes Weighted Cross-Entropy and Focal Loss to improve performance on underrepresented classes.
- **Data Augmentation**: Enhances dataset diversity with techniques such as random rotation, flipping, and brightness adjustments. 
- **Real-Time Segmentation**: Applies the trained model to laparoscopic videos, producing RGB mask overlays in real-time with OpenCV.
- **Evaluation Metrics**: Includes IoU, Dice Coefficient, and pixel-wise accuracy for a holistic evaluation of the model's performance.
- **Experiment Tracking**: Uses Weights and Biases to log experiments, track training progress, and visualize results.
