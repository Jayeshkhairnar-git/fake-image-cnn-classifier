# Fake Image CNN Classifier

A lightweight Convolutional Neural Network (CNN) built with PyTorch to classify images as **Real** or **Fake** (AI-generated). The model achieves **94% test accuracy** while staying under 100,000 trainable parameters.

## Highlights

- Custom CNN architecture with 4 convolutional blocks (16 → 32 → 64 → 128 channels)
- Binary classification: Real vs AI-generated/fake images
- Data augmentation: random horizontal flip, random rotation
- Model interpretability using **Grad-CAM** (visualises which image regions influence predictions)
- Feature space analysis using **PCA** on CNN-extracted features

## Tech Stack

| Component | Technology |
|---|---|
| Deep Learning | PyTorch, torchvision |
| Model Interpretability | Grad-CAM |
| Feature Analysis | PCA (scikit-learn) |
| Visualisation | Matplotlib |
| Data Loading | ImageFolder, DataLoader |

## Model Architecture

```
Input (32×32 RGB)
  → Conv Block 1: Conv2d → BatchNorm → ReLU → MaxPool  [16 channels]
  → Conv Block 2: Conv2d → BatchNorm → ReLU → MaxPool  [32 channels]
  → Conv Block 3: Conv2d → BatchNorm → ReLU → MaxPool  [64 channels]
  → Conv Block 4: Conv2d → BatchNorm → ReLU → MaxPool  [128 channels]
  → Global Average Pooling
  → Fully Connected → Output (1 neuron, BCEWithLogitsLoss)
```

**Total parameters: < 100,000**

## Training Details

| Parameter | Value |
|---|---|
| Epochs | 15 |
| Batch Size | 32 |
| Optimizer | Adam (lr=0.001) |
| Loss Function | BCEWithLogitsLoss |
| Device | CUDA / CPU (auto-detected) |

## Results

- **Test Accuracy: 94%**
- Balanced precision and recall for both classes
- Clear class separation visible in PCA feature space plot
- Grad-CAM confirms model focuses on semantically meaningful image regions

## Files

| File | Description |
|---|---|
| `Fake_Image_classifier_final__.ipynb` | Full training, evaluation, and Grad-CAM notebook |
| `fake_image_cnn.pth` | Saved model weights (PyTorch state_dict) |

## Course Context

Built as part of the **Data Science Lab** module at Hochschule Heilbronn (Winter Semester 2025/26).
