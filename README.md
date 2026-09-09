# 🌿 Explainable Plant Disease Classification with CNN and LIME

### CNN-Based Plant Disease Recognition Using PlantVillage with Explainable AI and User-Provided Image Prediction

![CNN](https://img.shields.io/badge/Model-CNN-blue)
![Keras](https://img.shields.io/badge/Framework-Keras-red)
![PlantVillage](https://img.shields.io/badge/Dataset-PlantVillage-green)
![Explainable AI](https://img.shields.io/badge/XAI-LIME-purple)
![OpenCV](https://img.shields.io/badge/Image%20Processing-OpenCV-orange)
![Agriculture AI](https://img.shields.io/badge/Domain-Precision%20Agriculture-brightgreen)
![Classification](https://img.shields.io/badge/Task-Plant%20Disease%20Classification-blueviolet)

This project presents an automated **plant disease classification framework** based on Convolutional Neural Networks (CNNs) for identifying plant diseases from leaf images.

The system is developed using images from the **PlantVillage** dataset and incorporates convolutional feature extraction, batch normalization, max pooling, dropout regularization, and fully connected layers.

Beyond predictive classification, the framework integrates **Local Interpretable Model-agnostic Explanations (LIME)** to provide visual explanations of model decisions.

A user-oriented prediction pipeline is also included, allowing external plant images to be loaded, preprocessed, classified, displayed for verification, and recorded for later reference.

---

# 🔎 Overview

Accurate and timely diagnosis of plant diseases is important for sustainable agriculture, crop management, and food security.

Traditional plant disease identification often depends on:

```text
Manual Inspection
      +
Expert Knowledge
      +
Time-Consuming Diagnosis
```

These limitations motivate the development of automated image-based disease recognition systems.

Deep learning, particularly **Convolutional Neural Networks**, provides an effective approach for automatically learning disease-related visual patterns directly from plant images.

The proposed workflow can be summarized as:

```text
Plant Leaf Image
       │
       ▼
Image Preprocessing
       │
       ▼
Data Augmentation
       │
       ▼
CNN Feature Extraction
       │
       ▼
Batch Normalization
       │
       ▼
Max Pooling
       │
       ▼
Dropout Regularization
       │
       ▼
Fully Connected Layers
       │
       ▼
Plant Disease Classification
       │
       ├──────────────► LIME Explanation
       │
       └──────────────► User Prediction Interface
```

---

# ✨ Main Objectives

The framework focuses on four major objectives:

```text
Accurate Plant Disease Classification
                 +
Robust CNN Feature Learning
                 +
Explainable Model Predictions
                 +
Practical User-Provided Image Prediction
```

The project aims to support automated plant disease analysis while improving transparency through explainable artificial intelligence.

---

# 🗂️ Dataset

The project uses the **PlantVillage** image dataset.

Dataset source:

```text
https://www.kaggle.com/datasets/arjuntejaswi/plant-village
```

The dataset contains images representing different:

```text
Plant Species
     +
Plant Diseases
     +
Healthy Plant Conditions
```

These images are used to train and evaluate the convolutional neural network for multi-class plant disease classification.

---

# 🧹 Data Preparation

Before model training, plant images are prepared for CNN-based classification.

The general preprocessing workflow is:

```text
Raw Plant Images
       │
       ▼
Image Loading
       │
       ▼
Image Resizing
       │
       ▼
Normalization
       │
       ▼
Data Augmentation
       │
       ▼
CNN Input
```

---

# 🔄 Data Augmentation

Data augmentation is incorporated into the training pipeline to increase image diversity and improve model generalization.

The purpose of augmentation is to expose the CNN to different visual variations of plant images during training.

Conceptually:

```text
Original Image
      │
      ▼
Transformation
      │
      ▼
Augmented Image
      │
      ▼
CNN Training
```

This helps reduce sensitivity to variations in image acquisition conditions.

---

# 🏗️ CNN Architecture

The plant disease classifier is constructed using the **Sequential API from Keras**.

The architecture includes:

```text
Input Image
    │
    ▼
Convolutional Layer
    │
    ▼
ReLU Activation
    │
    ▼
Batch Normalization
    │
    ▼
Max Pooling
    │
    ▼
Convolutional Feature Extraction
    │
    ▼
Dropout
    │
    ▼
Fully Connected Layers
    │
    ▼
Multi-Class Disease Prediction
```

The model combines several standard CNN components to learn discriminative representations from plant images.

---

# 🧩 Convolution Operation

Convolutional layers automatically learn local visual patterns associated with plant diseases.

The convolution operation is represented as:

```text
Conv(X, W)
=
Σᵢ Σⱼ X(i,j) × W(i,j) + b
```

where:

```text
X = input feature map
W = convolution kernel
b = bias
```

Convolution allows the model to detect local structures such as:

```text
Leaf Texture
Lesion Patterns
Discoloration
Surface Abnormalities
Disease-Related Visual Features
```

---

# ⚡ ReLU Activation

The CNN uses the **Rectified Linear Unit (ReLU)** activation function:

```text
ReLU(x) = max(0, x)
```

ReLU introduces non-linearity into the network while maintaining a simple computational structure.

---

# 📐 Batch Normalization

Batch normalization is incorporated to normalize intermediate activations.

The general formulation is:

```text
BatchNorm(x)
=
(x - μ)
────────── × γ + β
√(σ² + ε)
```

where:

```text
μ = batch mean
σ² = batch variance
γ = learned scaling parameter
β = learned shifting parameter
ε = numerical stability constant
```

Batch normalization helps stabilize intermediate feature distributions during training.

---

# 🔽 Max Pooling

Max pooling reduces spatial dimensionality by selecting the strongest activation from a local region.

```text
MaxPooling(X)
=
max X(i,j)
```

Conceptually:

```text
Feature Map
     │
     ▼
Local Regions
     │
     ▼
Maximum Activation
     │
     ▼
Reduced Feature Map
```

This reduces computational complexity while preserving prominent image features.

---

# 🛡️ Dropout Regularization

Dropout is incorporated to reduce overfitting.

The operation can be represented as:

```text
Dropout(X, p)
=
keep_probability × X
```

During training, selected activations are randomly suppressed.

This encourages the network to learn more distributed representations rather than relying heavily on individual neurons.

---

# ⚙️ Model Compilation

The CNN is compiled using:

```text
Optimizer     : Adam
Loss Function : Sparse Categorical Cross-Entropy
Task          : Multi-Class Classification
```

---

# 📉 Sparse Categorical Cross-Entropy

Sparse categorical cross-entropy is used to optimize the multi-class classification objective.

Conceptually:

```text
Loss
=
- (1/N)
Σ
log(
Predicted Probability of Correct Class
)
```

The objective encourages the network to assign higher probability to the correct plant disease class.

---

# 🚀 Adam Optimizer

The Adam optimizer is used for network optimization.

The general parameter update can be represented as:

```text
θ(t+1)
=
θ(t)
-
α × m_t
────────────
√v_t + ε
```

where:

```text
θ   = model parameters
α   = learning rate
m_t = first-moment estimate
v_t = second-moment estimate
ε   = numerical stability constant
```

---

# 🧠 Model Training

The CNN is trained using the prepared training images.

During training, the following quantities are monitored:

```text
Training Accuracy
Training Loss
Validation Accuracy
Validation Loss
```

The overall process is:

```text
Training Dataset
       │
       ▼
CNN Training
       │
       ▼
Update Parameters
       │
       ▼
Validation
       │
       ▼
Monitor Accuracy & Loss
       │
       ▼
Final Model
```

---

# 📊 Model Evaluation

After training, the model is evaluated using classification metrics including:

```text
Accuracy
Precision
Recall
Confusion Matrix
```

---

# ✅ Accuracy

Accuracy measures the proportion of correctly classified images.

```text
Accuracy
=
Number of Correct Predictions
─────────────────────────────
Total Number of Predictions
```

---

# 🎯 Precision

Precision measures how many samples predicted as a particular disease class are actually associated with that class.

Conceptually:

```text
Precision
=
TP
───────
TP + FP
```

---

# 🚨 Recall

Recall measures how many samples belonging to a disease class are correctly detected.

```text
Recall
=
TP
───────
TP + FN
```

---

# 🧮 Confusion Matrix

A confusion matrix is used to examine class-level prediction behavior.

```text
                  Predicted Class
                 ┌───────────────┐
Actual Class ───►│ Classification │
                 │    Matrix      │
                 └───────────────┘
```

It helps identify:

```text
Correct Predictions
False Positives
False Negatives
Frequently Confused Disease Classes
```

---

# 🔍 Explainable AI with LIME

One of the main components of the project is the integration of **LIME — Local Interpretable Model-agnostic Explanations**.

CNNs can provide strong predictive performance, but their decision process is often difficult to interpret.

LIME is used to explain individual plant disease predictions.

The general workflow is:

```text
Plant Image
     │
     ▼
CNN Prediction
     │
     ▼
Selected Disease Class
     │
     ▼
LIME
     │
     ▼
Local Image Perturbations
     │
     ▼
Prediction Sensitivity Analysis
     │
     ▼
Important Image Regions
     │
     ▼
Visual Explanation
```

---

# 💡 Why LIME?

Explainability is particularly important in precision agriculture because users may want to understand why a model predicts a particular disease.

LIME provides local explanations that can highlight regions of the image that contribute strongly to the model prediction.

This can help improve:

```text
Model Transparency
      +
Prediction Interpretability
      +
User Trust
```

---

# 🌿 Plant Disease Explanation

For a selected test image:

```text
Original Leaf Image
        │
        ▼
CNN Classification
        │
        ▼
Predicted Disease
        │
        ▼
LIME Explanation
        │
        ▼
Highlighted Influential Regions
```

The explanation provides visual insight into which regions of the plant image affected the classification decision.

---

# 📤 User-Provided Image Prediction

The trained CNN can also classify images provided directly by a user.

The prediction pipeline follows:

```text
User Provides Image Path
          │
          ▼
Load Image with OpenCV
          │
          ▼
Image Preprocessing
          │
          ▼
Normalization
          │
          ▼
Display Image
          │
          ▼
CNN Prediction
          │
          ▼
Disease Probability
          │
          ▼
Select Most Probable Class
          │
          ▼
Display Prediction
          │
          ▼
Save Result to Text File
```

---

# 🖼️ Image Loading with OpenCV

User-provided images are loaded using **OpenCV**.

The image is transformed into the format expected by the trained CNN.

The process includes:

```text
Load
 ↓
Preprocess
 ↓
Normalize
 ↓
Prepare Batch
 ↓
Model Prediction
```

---

# 👁️ User Verification

After preprocessing, the normalized image is displayed so that the user can verify that the correct image has been loaded.

This provides a simple visual check before interpreting the model prediction.

---

# 🔮 Disease Prediction

The CNN produces a probability distribution across the available plant disease classes.

Conceptually:

```text
Model Output
[
P(Class₁),
P(Class₂),
...
P(Classₙ)
]
```

The predicted disease is selected using the class with the highest probability:

```text
Predicted Class
=
argmax(Model Output)
```

---

# 💾 Prediction Storage

After classification, the result is:

```text
Displayed to the User
        +
Saved to a Text File
```

for later reference.

This provides a simple mechanism for retaining prediction records.

---

# 🌾 Practical Agricultural Application

The user-oriented prediction process is intended to support plant disease analysis in practical agricultural settings.

Potential users include:

```text
Farmers
Agricultural Practitioners
Crop Management Personnel
Precision Agriculture Researchers
```

A typical use case is:

```text
Capture Plant Image
        │
        ▼
Provide Image to System
        │
        ▼
Automated Disease Classification
        │
        ▼
Review Prediction
        │
        ▼
Inspect LIME Explanation
        │
        ▼
Support Early Crop Management Decision
```

---

# 🧪 Complete Experimental Workflow

The complete project pipeline can be summarized as:

```text
PlantVillage Dataset
        │
        ▼
Data Preparation
        │
        ▼
Data Augmentation
        │
        ▼
CNN Architecture
        │
        ├── Convolution
        ├── ReLU
        ├── Batch Normalization
        ├── Max Pooling
        └── Dropout
        │
        ▼
Model Compilation
        │
        ├── Adam
        └── Sparse Categorical Cross-Entropy
        │
        ▼
Training
        │
        ▼
Validation
        │
        ▼
Evaluation
        │
        ├── Accuracy
        ├── Precision
        ├── Recall
        └── Confusion Matrix
        │
        ├───────────────────┐
        │                   │
        ▼                   ▼
LIME Explanation      User Image Prediction
        │                   │
        └─────────┬─────────┘
                  ▼
          Explainable Plant
          Disease Diagnosis
```

---

# 🌍 Research Significance

The project combines:

```text
Deep Learning
      +
Plant Disease Recognition
      +
Explainable AI
      +
Practical User Interaction
```

The goal is not only to classify plant diseases but also to make the predictions more understandable through LIME-based visual explanations.

This interpretability component is particularly relevant in agricultural applications where transparent model behavior can support confidence in AI-assisted decision-making.

---

# 🎯 Intended Use

The framework is designed for research and experimentation in:

```text
Plant Disease Classification
Computer Vision
Deep Learning
Explainable Artificial Intelligence
Precision Agriculture
Agricultural Decision Support
Image-Based Disease Recognition
```

---

# 🚧 Scope

The current framework focuses on:

```text
PlantVillage Images
        +
CNN-Based Multi-Class Classification
        +
LIME Explainability
        +
User-Provided Image Prediction
```

The performance of the system should be evaluated using the actual trained model and corresponding held-out image data before making claims regarding practical field performance.


---

## 🌿 Explainable Plant Disease Classification

**Combining CNN-based visual recognition, PlantVillage data, LIME explainability, and user-provided image prediction for automated plant disease analysis.**
