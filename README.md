With the global demand for sustainable agriculture and food
security, the accurate and timely diagnosis of plant diseases has
become paramount. Traditional methods of disease detection
are often time-consuming and labor-intensive, necessitating the
need for advanced technological solutions. In recent years,
deep learning techniques, particularly Convolutional Neural
Networks (CNNs), have demonstrated remarkable success in
various image classification tasks, including medical diagnosis
and object recognition.
The critical challenge of automated
plant disease classification using a CNN-based approach. A rich dataset obtained from the PlantVillage
repository, encompassing diverse plant species and diseases.
The proposed model incorporates essential components such
as data augmentation, batch normalization, and dropout layers,
optimizing its ability to generalize patterns from images of
varying conditions.
The significance of our work lies not only in achieving
high classification accuracy but also in enhancing model inter-
pretability. To address the interpretability concern associated
with deep neural networks, we integrate Local Interpretable
Model-agnostic Explanations (LIME), offering insights into
the decision-making process of the CNN. This interpretability
aspect is crucial for building trust in the model’s predictions,
particularly in applications such as precision agriculture, where
transparent decision-making is essential.
Moreover, we explore the practical deployment of the model
through a user-friendly interface, allowing real-time predic-
tions from user-provided images. This feature opens avenues
for farmers and agricultural practitioners to employ the system
for on-field disease detection, contributing to early intervention
and improved crop management.





METHODOLOGY


A. Model Training and Model Evaluation
Train the model using the training set and validate it using the
testing set. Monitor the training and validation accuracy and
loss. Evaluate the final model on the validation set and report
key metrics such as accuracy, precision, recall, and confusion
matrix.
1https://www.kaggle.com/datasets/arjuntejaswi/plant-village


B. Model Architecture
Construct a CNN model using the Sequential API from
Keras. The architecture includes convolutional layers, acti-
vation functions (ReLU), batch normalization, max-pooling
layers, dropout layers for regularization, and fully connected
layers.
Convolution operation
Conv(X, W ) =
m∑
i=1
n∑
j=1
X(i,j) · W(i,j) + b
Rectified Linear Unit (ReLU) activation
ReLU(x) = max(0, x)
Batch Normalization
BatchNorm(x) = x − μ
√σ2 + ε · γ + β
Max-Pooling
MaxPooling(X) = max
i,j (X(i,j))
Dropout
Dropout(X, p) = keep prob × X


C. Model Compilation
Compile the Adam optimizer and sparse categorical crossen-
tropy loss function for multiclass classification.
SparseCategoricalCrossentropy(y, ˆy) = − 1
N
N∑
i=1
yi·log(ˆyi)+(1−yi)·log(1−
Adam Optimizer Update Rule
θt+1 = θt − α · mt
√vt + ε
Accuracy = Number of Correct Predictions
Total Number of Predictions


D. Explainability Analysis
Use LIME (Local Interpretable Model-agnostic Explanations)
to explain the model predictions for a sample image from the
test set.


E. Prediction of User-Provided Images
Once the convolutional neural network (CNN) model is
trained, it can be employed to predict plant diseases for user-
provided images. The following steps illustrate the process.
Here the users are prompted to input an image file path con-
taining a plant for disease classification. The system loads and
preprocesses the image using OpenCV, ensuring compatibility
with the trained convolutional neural network (CNN) model.
The normalized image is then displayed for user verification.
The model predicts the plant disease class by analyzing the
preprocessed image, and the most probable class is selected.
The result is output to the console and stored in a text file for
reference. This user-friendly process empowers stakeholders
to quickly assess potential plant diseases and make informed
decisions in agricultural contexts
