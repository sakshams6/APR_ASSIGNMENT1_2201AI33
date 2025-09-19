# Iris Flower Classification using Support Vector Machine (SVM) 🌸

This project is a Python script that demonstrates a classic machine learning classification task. It uses a Support Vector Machine (SVM) algorithm to predict the species of an Iris flower based on its petal length and width.

The script trains the model, evaluates its performance, and generates two key visualizations: a decision boundary plot and a confusion matrix.

## Features

* **Algorithm:** Support Vector Machine (SVC) with a linear kernel.
* **Dataset:** The classic Iris dataset, loaded directly from `scikit-learn`.
* **Train-Test Split:** The data is split into 60% for training and 40% for testing to ensure a robust evaluation.
* **Stratified Sampling:** The split is stratified, meaning the proportion of each flower species is the same in both the training and testing sets. This leads to a more balanced and reliable model evaluation.
* **Visualizations:**
    1.  **Decision Boundary Plot:** A 2D plot showing the regions where the model will classify a flower as Setosa, Versicolor, or Virginica.
    2.  **Confusion Matrix:** A heatmap that visually represents the model's performance, with numbers manually written in each block for clarity.

---

## How to Run

Follow these steps to run the project on your local machine.

### 1. Prerequisites

Make sure you have Python 3 installed on your system.

### 2. Clone or Download

Download the `main_script_final_with_both_plots.py` file and place it in a new folder on your computer.

### 3. Install Dependencies

Open your terminal or command prompt, navigate to the project folder, and install the required Python libraries by running the following command:

```bash
pip install numpy scikit-learn matplotlib seaborn
