# Image Classification: Cats vs. Dogs using SVM

This project implements an image classification model using **Support Vector Machines (SVM)** to classify images of cats and dogs. The model uses a subset of the **Kaggle Cats vs. Dogs dataset**, with a **80/20 train-test split**.

## Project Overview

The primary goal of this project is to classify images of cats and dogs using an SVM model with an RBF kernel. The dataset used is a small portion of the Kaggle **Cats vs. Dogs** dataset, which is manually selected, and the model was trained using **80% of the data** and tested on **20%**.

### Dataset
- The original dataset can be found on [Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data).
- This project uses **only a subset of the Kaggle dataset**. The images were divided manually into training and test folders:
  - **Training Set**: 80% of the images
  - **Test Set**: 20% of the images

### Model
- **Model Type**: Support Vector Machine (SVM)
- **Kernel**: RBF (Radial Basis Function)
- **Image Preprocessing**: 
  - Images are resized to **64x64 pixels**.
  - Each image is flattened into a **1D vector**.
  - Images are normalized and scaled using **StandardScaler**.
  
### Workflow

1. **Image Loading**: The images are loaded from the local filesystem using `OpenCV`. Training images are labeled based on the filename (i.e., "cat" or "dog").
2. **Train-Test Split**: The data is split manually into 80% for training and 20% for testing.
3. **Data Preprocessing**: The images are resized, flattened, and normalized.
4. **Model Training**: An SVM classifier with an RBF kernel is trained on the scaled training data.
5. **Evaluation**: The model is evaluated using various metrics like **accuracy**, **precision**, **recall**, and **F1-score** on the test set.

## File Structure

```bash
├── new_train/                 # Folder containing training images (cats and dogs)
├── new_test/                  # Folder containing test images (cats and dogs)
├── model_code.ipynb           # Jupyter Notebook containing the model implementation
├── README.md                  # Project README file
└── requirements.txt           # Required dependencies
```

## Setup Instructions

### Prerequisites
- **Python 3.x**
- **Jupyter Notebook** or any Python IDE
- **Libraries**:
  - `numpy`
  - `opencv-python`
  - `scikit-learn`

You can install the required libraries using the following command:

```bash
pip install -r requirements.txt
```

### Run the Model
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   ```
2. Navigate to the project folder:
   ```bash
   cd your-repo-name
   ```
3. Open the `model_code.ipynb` Jupyter Notebook and run each cell sequentially to train the model and evaluate it on the test set.

### Model Training

- The model is trained using **80% of the dataset**. This subset is manually curated from the Kaggle dataset.
- The remaining **20% of the dataset** is used for testing.

### Evaluation Metrics

Once the model is trained, it is evaluated using the following metrics:
- **Accuracy**: Overall percentage of correctly classified images.
- **Confusion Matrix**: Shows the breakdown of true positives, true negatives, false positives, and false negatives.
- **Precision, Recall, and F1-Score**: For both cats and dogs, giving a deeper insight into the classification performance.

## Limitations & Future Work

- **Limited Dataset**: This project only uses a portion of the Kaggle dataset, which limits its generalizability.
- **SVM for Image Classification**: SVM is not typically used for large-scale image classification tasks. In future iterations, we can explore using **Convolutional Neural Networks (CNNs)** for better performance.
- **Data Augmentation**: Adding data augmentation techniques such as flipping, rotation, and zooming could further improve model performance.

                                                                                                                                                                                                        
  
  
  
  
  
  
  
  
  
  
