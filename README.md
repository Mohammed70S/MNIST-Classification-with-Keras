
# Fashion MNIST Classification with Keras

This project uses a neural network model built with Keras to classify images of clothing items from the Fashion MNIST dataset. The aim is to train an artificial neural network (ANN) that can accurately categorize images into one of 10 distinct fashion classes.

## Project Objective

The objective is to design and train a multiclass classification model using the Fashion MNIST dataset. We use neural networks with multiple dense layers and regularization techniques to optimize model performance, targeting an accuracy of at least 90% on the test set.

## Dataset Overview

The [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) contains 70,000 grayscale images, each 28x28 pixels, in 10 clothing categories:
- **Training Set**: 60,000 images
- **Test Set**: 10,000 images

Each image belongs to one of the following classes:
1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

## Instructions for Running the Code

1. **Installation**: Install the required dependencies (see below).
2. **Run the Notebook**: Open and execute the notebook cell by cell to train the model, visualize the results, and evaluate performance.
3. **Results**: The notebook will output model performance metrics, classification reports, and confusion matrix visualizations.

### Dependencies and Installation Instructions

This project requires Python with the following libraries:
- **TensorFlow** for model building and training.
- **scikit-learn** for model validation and evaluation.
- **matplotlib** and **seaborn** for data visualization.

Install dependencies via pip:
```bash
pip install numpy tensorflow scikit-learn matplotlib seaborn
```

## Code Description

### Preprocessing

- **Normalization**: Scales pixel values between 0 and 1 for more efficient model training.
- **Reshaping**: Converts each 28x28 image to a 784-element vector for compatibility with the ANN model.
- **Train-Validation Split**: Splits the original training data (80% for training, 20% for validation).

### Model Architecture

The model is a feedforward neural network with the following layers:
1. **Input Layer**: Accepts the flattened 784-dimensional input.
2. **Hidden Layers**: Three fully connected layers of varying neuron counts (256, 128, 64) with LeakyReLU activation, BatchNormalization, and Dropout for regularization.
3. **Output Layer**: A softmax layer with 10 neurons, representing each fashion class.

### Model Training and Regularization

- **Loss Function**: `sparse_categorical_crossentropy` for multiclass classification.
- **Optimizer**: `Adam` optimizer.
- **Learning Rate Scheduler**: Gradually decreases the learning rate for stable training.
- **Early Stopping**: Halts training if validation performance stagnates, preventing overfitting.

### Evaluation

The notebook includes:
- **Test Accuracy**: Performance on the test set.
- **Classification Report**: Precision, recall, and F1-score for each class.
- **Confusion Matrix**: Visualized with a heatmap to illustrate model predictions vs. actual labels.
- **Training and Validation Curves**: Loss and accuracy curves across epochs.

## Expected Results

After training, the model should:
- Achieve a test accuracy close to or above 90%.
- Provide a detailed classification report with high precision and recall for each class.
- Display loss and accuracy curves that show consistent training and minimal overfitting.

---

This README provides an overview and instructions for running the project, helping users understand the purpose, approach, and expected outcomes of the Fashion MNIST classification. Let me know if you’d like any adjustments!
