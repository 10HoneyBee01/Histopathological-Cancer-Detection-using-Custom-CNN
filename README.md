![Histopathological Cancer](https://github.com/10HoneyBee01/Histopathological-Cancer-Detection-using-Custom-CNN/blob/main/design.PNG)
# Histopathological Cancer Detection using 5-Layer CNN Model

🔬🩺🔍📷🧠

This repository contains a deep learning model for histopathological cancer detection using a 5-layer Convolutional Neural Network (CNN). The model is trained on the histopathological images dataset available on Kaggle: [Histopathologic Cancer Detection Competition](https://www.kaggle.com/competitions/histopathologic-cancer-detection).

## Introduction

Histopathological analysis plays a crucial role in the diagnosis and treatment of cancer. With the advancements in deep learning techniques, CNNs have shown remarkable performance in automated cancer detection from histopathological images. This repository presents an implementation of a 5-layer CNN model trained for detecting cancerous regions in histopathological images.

## Dataset

📊📉📈🖼️

The dataset used for training this model is provided by the Histopathologic Cancer Detection Competition hosted on Kaggle. The dataset contains a large collection of histopathological images of lymph node sections, labeled as either positive (cancerous) or negative (non-cancerous).

You can download the dataset from the following link: [Histopathologic Cancer Detection Dataset](https://www.kaggle.com/competitions/histopathologic-cancer-detection)

## Model Architecture

🏗️🧱🛠️

The CNN model architecture used for cancer detection consists of five convolutional layers followed by batch normalization, ReLU activation, and max-pooling layers. The final layers include fully connected layers for classification.

## How to Use

🚀🎯🛠️

1. **Dataset Preparation**:
   - Download the dataset from the provided link.
   - Organize the dataset into appropriate directories, separating images into 'positive' and 'negative' classes.

2. **Training**:
   - Run the training script provided in this repository after adjusting hyperparameters and file paths as necessary.
   - The training script will train the CNN model on the provided dataset.

3. **Evaluation**:
   - Evaluate the trained model's performance using evaluation metrics such as accuracy, precision, recall, and F1-score.
   - Use the model to make predictions on unseen histopathological images and analyze the results.

## Requirements

🔧📋🛠️

- Python 3.x
- PyTorch
- torchvision
- NumPy
- pandas
- matplotlib
- scikit-learn

## Contribution

🤝👥🛠️

Contributions to improving the model architecture, dataset processing, or any other aspect of the project are welcome. If you find any issues or have suggestions for enhancements, please feel free to open an issue or submit a pull request.

## References

📚🔖📖🔍

- Dataset: [Histopathologic Cancer Detection Competition](https://www.kaggle.com/competitions/histopathologic-cancer-detection)
- PyTorch Documentation: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
- Deep Learning for Histopathological Cancer Detection: A Survey - [https://arxiv.org/abs/2102.04446](https://arxiv.org/abs/2102.04446)

---

*Note: This project is for educational and research purposes only. It is not intended for clinical use.*
