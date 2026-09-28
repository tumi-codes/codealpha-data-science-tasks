# Iris Flower Classification Using Logistic Regression

## 📌 Project Overview

This project uses **Machine Learning to classify Iris flowers into different species** based on their physical measurements.

The project uses the Iris dataset and implements a **Logistic Regression** classification model with Python's `pandas` and `scikit-learn` libraries.

The workflow includes data loading, exploratory data analysis, data preprocessing, model training, prediction, and model accuracy evaluation.

---

## 🎯 Project Objectives

The main objectives of this project are to:

* Load and explore the Iris dataset using Pandas.
* Examine the dataset's statistical properties and check for missing values.
* Preprocess the dataset by removing unnecessary columns and encoding categorical variables.
* Split the dataset into training and testing sets.
* Train a Logistic Regression model to classify Iris flower species.
* Generate predictions on unseen test data.
* Evaluate the model using classification accuracy.

---

## 📂 Project Structure

```text
Iris-Flower-Classification/
│
├── analysis.ipynb      # Jupyter Notebook containing the complete analysis
├── Iris.csv            # Dataset used for training and testing
├── README.md            # Project documentation
└── requirements.txt     # Project dependencies
```

---

## 📊 Dataset Description

The project uses the **Iris dataset**, which contains measurements of Iris flowers belonging to three species.

### Features

| Column          | Description                                     |
| --------------- | ----------------------------------------------- |
| `Id`            | Unique identifier for each flower               |
| `SepalLengthCm` | Sepal length in centimeters                     |
| `SepalWidthCm`  | Sepal width in centimeters                      |
| `PetalLengthCm` | Petal length in centimeters                     |
| `PetalWidthCm`  | Petal width in centimeters                      |
| `Species`       | Target variable representing the flower species |

### Target Variable

The `Species` column represents the flower species that the model is trained to predict.

The three species in the standard Iris dataset are:

* Iris-setosa
* Iris-versicolor
* Iris-virginica

---

## 🛠️ Technologies and Libraries

The project is implemented using the following tools:

| Technology       | Purpose                                           |
| ---------------- | ------------------------------------------------- |
| Python           | Main programming language                         |
| Pandas           | Data loading and manipulation                     |
| Scikit-learn     | Model training, dataset splitting, and evaluation |
| Jupyter Notebook | Interactive development and analysis              |

---

## ⚙️ Project Workflow

### 1. Importing the Libraries

The project begins by importing the necessary libraries.

```python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from pathlib import Path
```

These libraries are used for data manipulation, splitting the dataset, training the classification model, and working with file paths.

### 2. Loading the Dataset

The Iris dataset is loaded from a CSV file using Pandas.

```python
iris = pd.read_csv('Iris.csv')
```

The first five rows are displayed to inspect the dataset, while `describe()` provides descriptive statistics for the numerical columns.

```python
print(iris.head(5))
print(iris.describe())
```

### 3. Checking for Missing Values

The project checks for missing values in each column.

```python
print(iris.isna().sum())
```

This helps identify whether any columns contain missing data that could affect model training.

### 4. Data Preprocessing

The dataset is prepared for machine learning by removing the `Id` column and converting the target variable into numerical codes.

```python
iris.drop('Id', axis=1, inplace=True)

iris['Species'] = iris['Species'].astype('category')
iris['Species'] = iris['Species'].cat.codes
```

**Preprocessing steps:**

* The `Id` column is removed because it is not needed for the classification task.
* The `Species` column is converted to a categorical data type.
* The categorical species labels are encoded into numerical values so they can be used as the target variable.

### 5. Splitting the Dataset

The dataset is divided into training and testing sets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    iris.drop('Species', axis=1),
    iris['Species'],
    test_size=0.2,
    random_state=42
)
```

**Split configuration:**

| Parameter       | Value                    |
| --------------- | ------------------------ |
| Training data   | 80%                      |
| Testing data    | 20%                      |
| Random state    | 42                       |
| Input features  | Four flower measurements |
| Target variable | Species                  |

The training set is used to train the model, while the testing set is used to evaluate its performance on data that was not used during training.

### 6. Model Training

The project uses Logistic Regression as its classification algorithm.

```python
logistic_model = LogisticRegression(random_state=42)

logistic_model.fit(X_train, y_train)
```

The model learns the relationship between the flower measurements and their corresponding species labels.

### 7. Making Predictions

After training, the model predicts the species of flowers in the test dataset.

```python
prediction = logistic_model.predict(X_test)
```

The predictions are stored in the `prediction` variable.

### 8. Model Evaluation

The model's performance is evaluated using its accuracy score on the test dataset.

```python
model_accuracy = logistic_model.score(X_test, y_test)

print(f"Model Accuracy: {model_accuracy}")
```

The accuracy score represents the proportion of test samples that the model classified correctly.

---

## 📈 Model Evaluation Metric

### Accuracy

Accuracy is calculated as:

$$
\text{Accuracy} =
\frac{\text{Correct Predictions}}{\text{Total Predictions}}
$$

The score ranges from 0 to 1, with higher values indicating that more test samples were classified correctly.

The actual accuracy is generated when the notebook is executed.

---

## 🚀 Installation and Setup

### Prerequisites

Ensure that you have the following installed:

* Python 3.9 or later
* Jupyter Notebook or JupyterLab
* pip

### Step 1: Clone the Repository

```bash
git clone <your-repository-url>
cd Iris-Flower-Classification
```

Replace `<your-repository-url>` with the URL of your GitHub repository.

### Step 2: Install Dependencies

Install the required libraries:

```bash
pip install pandas scikit-learn jupyter
```

### Step 3: Add the Dataset

Ensure that `Iris.csv` is located in the same directory as `analysis.ipynb`.

### Step 4: Launch Jupyter Notebook

```bash
jupyter notebook
```

Open `analysis.ipynb` and execute the cells sequentially.

---

## 📦 Requirements

Create a `requirements.txt` file containing:

```text
pandas
scikit-learn
jupyter
```

Install the dependencies with:

```bash
pip install -r requirements.txt
```

---

## 💡 Key Learnings

Through this project, I explored:

* Loading and inspecting datasets with Pandas.
* Checking for missing values and understanding descriptive statistics.
* Preparing categorical target variables for machine learning.
* Splitting data into training and testing sets.
* Training a Logistic Regression classification model.
* Generating predictions using a trained model.
* Evaluating classification performance using accuracy.

---

## 🔮 Future Improvements

Potential improvements to this project include:

* Adding visualizations to explore relationships between flower measurements.
* Evaluating the model using a confusion matrix and classification report.
* Comparing Logistic Regression with other classification algorithms, such as Decision Trees, K-Nearest Neighbors, and Support Vector Machines.
* Using stratified train-test splitting to preserve class proportions.
* Testing the model with new flower measurements.
* Saving the trained model for reuse.

---

## 👤 Author

**Tumininu Akintola**

This project was developed as part of my journey into Data Science and Machine Learning.

---

## 📄 License

This project is intended for educational and learning purposes. Add a license file if you plan to distribute the project under specific terms.
