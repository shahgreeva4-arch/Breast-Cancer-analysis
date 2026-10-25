# Breast Cancer Analysis

Classifying breast tumours as **malignant** or **benign** with an Artificial Neural Network (ANN), using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

## Contents

| File | Description |
|------|-------------|
| `Breast_cancer.ipynb` | Main notebook: data cleaning, visualisation, model training and evaluation |
| `wdbc.data` / `wdbc.names` | Wisconsin Diagnostic Breast Cancer dataset (used by the notebook) and its description |
| `wpbc.data` / `wpbc.names` | Wisconsin Prognostic Breast Cancer dataset and its description |
| `breast-cancer-wisconsin.data` / `.names` | Original Wisconsin breast cancer dataset and its description |
| `unformatted-data` | Raw, unformatted version of the original dataset |
| `Index` | Index of the dataset files |

## Workflow

1. **Data preparation** – drop the `id` column and encode the diagnosis (Malignant = 1, Benign = 0); check for missing values.
2. **Exploration** – plot the class distribution and a correlation heatmap of the mean features.
3. **Preprocessing** – stratified 80/20 train-test split and feature scaling with `StandardScaler`.
4. **Model** – Keras `Sequential` network with Dense layers and Dropout, wrapped with SciKeras `KerasClassifier`.
5. **Tuning** – `GridSearchCV` over dropout rate and learning rate, with `EarlyStopping` to prevent overfitting.
6. **Evaluation** – classification report, confusion matrix and ROC-AUC on the held-out test set.

## Results

The tuned model reaches about **98% accuracy** on the unseen test data. The notebook ends with a plain-English performance report that also looks at precision and recall, since missing a malignant case matters most in a medical setting.

## Requirements

Python 3 with `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `tensorflow` and `scikeras`.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow scikeras
```

Then open `Breast_cancer.ipynb` in Jupyter and run all cells.

## Dataset source

UCI Machine Learning Repository – Breast Cancer Wisconsin datasets.
