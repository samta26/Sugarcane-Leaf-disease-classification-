🌱 **Sugarcane Leaf Disease Classification with InceptionV3**

**I. Overview**

This repository contains a deep learning project for sugarcane leaf disease classification using the InceptionV3 Convolutional Neural Network (CNN) architecture.The project leverages transfer learning on the ImageNet pre-trained InceptionV3 model, fine-tuning it to detect five major sugarcane leaf conditions:

1. Healthy
2. Mosaic
3. Red Rot
4. Rust
5. Yellow 

This system aims to support precision agriculture by providing fast, automated detection of sugarcane diseases that significantly affect crop yield.

**II. Dataset**

Source: Sugarcane Leaf Disease Dataset (https://www.kaggle.com/datasets/nirmalsankalana/sugarcane-leaf-disease-dataset)

Total images: 2521

Classes: 5 (Healthy, Mosaic, Red Rot, Rust, Yellow Leaf)

Distribution: Balanced across classes (≈500 images per class)

Format: RGB images, variable sizes

Preprocessing: Resized to 299×299×3 and normalized to [0,1]

**III. Learning Approach Used**

**Supervised Learning**

Since the dataset contains explicit labels for each image, a supervised CNN approach was used. The model learns visual patterns (color, spots, lesions, texture) directly from images to classify leaf health conditions.

Data Augmentation was applied to increase robustness:
- Rotation
- Horizontal/Vertical Flips
- Zooming
- Brightness adjustment

**IV. Algorithm Used**

**Base Model: InceptionV3 (Transfer Learning)**

1. Pre-trained on ImageNet
2. Top classification layers removed (include_top=False)
3. Custom layers added:
   - Global Average Pooling
   - Dense (128 units, ReLU)
   - Dropout (0.4 and 0.3 for regularization)
   - Dense (5 units, Softmax output)
4. Optimizer: Adam (lr=0.0001)Loss Function: Sparse Categorical Cross-EntropyTraining Epochs: 50Train-Test Split: 80:20

**V. Results & Evaluation**

The model achieved an overall accuracy of 79% with a macro F1-score of 0.79. The ROC-AUC values were greater than 0.94 across all classes, with Rust achieving the highest at 0.98. The model also showed a strong Matthews Correlation Coefficient (MCC) of 0.7378, confirming its balanced predictive power across classes.

Key Insights:

- The Rust class had the best performance, with an F1-score of 0.89 and highest recall.
- Red Rot was the most challenging to classify due to overlapping features with Mosaic and Yellow, resulting in lower recall.
- The performance was fairly balanced across other disease categories, proving the model’s general reliability.
