**🧠 CNN on MNIST: Feature Extraction, KNN, and PCA Analysis**

A machine learning project exploring CNN-based feature learning, K-Nearest Neighbors classification, and PCA visualization on the MNIST dataset.
By Michelle Chala

**📌 Overview**

This project trains a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset, then extracts features from the trained CNN to:

Evaluate downstream classification using K-Nearest Neighbors (kNN)
Visualize learned representations with Principal Component Analysis (PCA)
The workflow combines supervised deep learning with unsupervised dimensionality reduction, offering insight into both classification performance and internal feature structure.

**🛠️ Tools & Technologies**

Python, NumPy, Pandas
TensorFlow / Keras
Matplotlib, Seaborn
Scikit-learn (for kNN & PCA)
📈 CNN Architecture & Training

Input: 28x28 grayscale images
Conv1: 8 filters, 3×3 kernel, ReLU, same padding
Pooling
Conv2: 16 filters, 3×3 kernel, ReLU
Pooling
Flatten → Dense (10 classes)
Optimizer: Adam
Loss: Sparse categorical crossentropy
Epochs: 100
Train Accuracy: ~99.6%
Test Accuracy: ~98.27%
🧪 Evaluation & Analysis

**🔍 Confusion Matrix**

Strong diagonal dominance
Most errors occurred between similar digits (e.g., 3s vs 2s, 6s vs 5s)
![image](https://github.com/user-attachments/assets/b9e15eab-d723-44fa-be19-a88b05729403)

**🔗 Feature Extraction**

784-dimensional feature vectors extracted post-flattening
kNN (k=5) on CNN features: ~98% accuracy
Confirms CNN learned highly discriminative representations

**🌀 PCA Visualization**

Reduced to 2D for plotting: clear digit clusters with minimal overlap
Reduced to 10D for classification:
kNN accuracy: ~95.7%
Slight performance drop due to information compression, but still strong
![image](https://github.com/user-attachments/assets/196c3120-7c1c-48dc-beaa-57776f252ae5)


**🎯 Key Takeaways**

CNNs not only perform well in classification but generate high-quality latent features
Even simple models (like kNN) can succeed when built on well-learned representations
PCA offers an intuitive visual confirmation of digit separability in feature space

**🙋‍♀️ Author**

Michelle Chala
Undergraduate in Computer Science & Psychology, Concentration in Computational Modelling & Cognitive Neuroscience
Focus: AI, Machine Learning, Cognitive Neuroscience, Quantum Computing
