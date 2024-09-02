# Food Recognition and Nutrition Analysis

## Overview

This project focuses on developing a Convolutional Neural Network (CNN) model to recognize and classify various food items from images. The model is built using TensorFlow and Keras and is trained on a dataset of food images to classify them into 20 different categories.

## Dataset

The dataset used in this project is a collection of food images categorized into 20 different classes. The dataset is structured as follows:

- **Path:** `C:/Users/badal/OneDrive/Desktop/ML And DL/ML_AND_DL/My_Projects/Food_recognition_and_Nutrition_Analysis/Food_Data`
- **Classes:**
  - burger
  - butter_naan
  - chai
  - chapati
  - chole_bhature
  - dal_makhani
  - dhokla
  - fried_rice
  - idli
  - jalebi
  - kaathi_rolls
  - kadai_paneer
  - kulfi
  - masala_dosa
  - momos
  - paani_puri
  - pakode
  - pav_bhaji
  - pizza
  - samosa

## Requirements

- TensorFlow 2.x
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

## Setup

1. Install the required libraries:
    ```bash
    pip install tensorflow numpy matplotlib seaborn scikit-learn
    ```

2. Ensure the dataset is properly organized in the specified directory.

## Model Architecture

The CNN model architecture is as follows:

1. **Block 1:**
   - Conv2D (64 filters, 3x3 kernel, ReLU activation)
   - Conv2D (64 filters, 3x3 kernel, ReLU activation)
   - MaxPooling2D (2x2 pool size)

2. **Block 2:**
   - Conv2D (128 filters, 3x3 kernel, ReLU activation)
   - Conv2D (128 filters, 3x3 kernel, ReLU activation)
   - MaxPooling2D (2x2 pool size)

3. **Block 3:**
   - Conv2D (256 filters, 3x3 kernel, ReLU activation)
   - Conv2D (256 filters, 3x3 kernel, ReLU activation)
   - Conv2D (256 filters, 3x3 kernel, ReLU activation)
   - MaxPooling2D (2x2 pool size)

4. **Block 4:**
   - Conv2D (512 filters, 3x3 kernel, ReLU activation)
   - Conv2D (512 filters, 3x3 kernel, ReLU activation)
   - Conv2D (512 filters, 3x3 kernel, ReLU activation)
   - MaxPooling2D (2x2 pool size)

5. **Block 5:**
   - Conv2D (512 filters, 3x3 kernel, ReLU activation)
   - Conv2D (512 filters, 3x3 kernel, ReLU activation)
   - Conv2D (512 filters, 3x3 kernel, ReLU activation)
   - MaxPooling2D (2x2 pool size)

6. **Fully Connected Layers:**
   - Flatten
   - Dense (20 units, softmax activation)

## Training

The model is trained with the following parameters:

- **Batch Size:** 64
- **Epochs:** 20
- **Optimizer:** Adam
- **Loss Function:** SparseCategoricalCrossentropy
- **Metrics:** Accuracy

Training and validation accuracy and loss are plotted to visualize performance.

## Evaluation

The model's performance is evaluated using a test set. The metrics include accuracy and a confusion matrix. The classification report provides precision, recall, and F1-score for each class.

## Visualization

- Training and validation accuracy and loss are plotted over epochs.
- A confusion matrix heatmap is generated to show the classification performance.

## Usage

To run the code, ensure that the dataset path is correctly specified and execute the Python script. The training process will output training and validation metrics, and the final evaluation metrics will be displayed.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- TensorFlow and Keras for the deep learning framework.
- The dataset used for training and evaluation.

