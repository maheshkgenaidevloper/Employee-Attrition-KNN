# Employee-Attrition-KNN
# 👥 Employee Attrition Prediction using KNN

<h1 align="center">🚀 Employee Attrition Prediction</h1>

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=25&pause=1000&color=36BCF7&center=true&vCenter=true&width=650&lines=Machine+Learning+Project;Powered+by+K-Nearest+Neighbors;Python+%7C+Scikit-learn+%7C+Streamlit" alt="Animated project introduction" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Machine-Learning-orange?logo=scikitlearn&logoColor=white" alt="Machine Learning"/>
  <img src="https://img.shields.io/badge/Algorithm-KNN-brightgreen" alt="KNN"/>
  <img src="https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/Status-Portfolio%20Project-success" alt="Project status"/>
</p>

<p align="center">
  <b>📊 Predict. Analyze. Learn.</b><br/>
  An interactive Machine Learning project demonstrating employee attrition classification using the K-Nearest Neighbors algorithm.
</p>

---

## 🎬 Project Highlights

<p align="center">
  <img src="https://media.giphy.com/media/3oKIPEqDGUULpEU0aQ/giphy.gif" width="300" alt="Animated data visualization" />
</p>

> 💡 **Project objective:** Build a machine learning application that classifies sample employee records into attrition categories and displays predictions through an interactive Streamlit dashboard.

## ✨ Features

- 🤖 **KNN Classification** — Predicts the attrition class from employee attributes.
- 📂 **CSV Dataset** — Loads sample data directly without a data-generation script.
- ⚙️ **Data Preprocessing** — Standardizes numerical features and encodes categorical data.
- 📈 **Model Evaluation** — Displays accuracy, precision, recall, F1-score, and confusion matrix.
- 🖥️ **Interactive Dashboard** — Explores sample data through Streamlit.
- 🔍 **Prediction Interface** — Enter example employee details to get a model prediction.
- 💾 **Model Persistence** — Saves the trained pipeline using Joblib.
- 🐙 **GitHub Ready** — Includes project documentation and dependency management.

## 🧰 Tech Stack

<p align="center">
  <img src="https://skillicons.dev/icons?i=python,sklearn,git,github,vscode" alt="Technology icons"/>
</p>

| Technology | Purpose |
|---|---|
| 🐍 Python | Core programming |
| 🧮 Pandas | Data manipulation |
| 🤖 Scikit-learn | KNN and model preprocessing |
| 📊 Streamlit | Interactive dashboard |
| 💾 Joblib | Saving and loading the trained model |
| 🐙 Git & GitHub | Version control and project hosting |

## 📁 Project Structure

```text
Employee-Attrition-KNN/
│
├── 📄 app.py
├── 🧠 train_model.py
├── 📊 employee_attrition.csv
├── 📦 requirements.txt
├── 📘 README.md
├── 🚫 .gitignore
│
└── 📂 models/
    └── knn_model.pkl
```

*Note: `knn_model.pkl` is created after running the training script.*

## 📊 Dataset Overview

The sample CSV contains these columns:

| Feature | Description |
|---|---|
| `Age` | Employee age |
| `MonthlyIncome` | Monthly income |
| `JobSatisfaction` | Satisfaction rating from 1 to 4 |
| `YearsAtCompany` | Years at the company |
| `WorkLifeBalance` | Work-life balance rating from 1 to 4 |
| `OverTime` | Overtime status |
| `Attrition` | Target label: `Yes` or `No` |

⚠️ The included dataset is synthetic and intended for educational demonstrations. It is not representative of a real workforce.

## 🧠 How KNN Works

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/e/e7/KnnClassification.svg" width="480" alt="K-nearest neighbors classification illustration"/>
</p>

The K-Nearest Neighbors algorithm classifies a new observation based on the labels of its closest observations in the training dataset.

**Workflow:**

1. 📥 Load employee data from CSV.
2. 🧹 Preprocess numerical and categorical features.
3. ✂️ Split the dataset into training and test sets.
4. 📏 Standardize numerical features.
5. 🤖 Train KNN with `k = 3`.
6. 📈 Evaluate the model on held-out test data.
7. 💾 Save the trained pipeline.
8. 🖥️ Display example predictions in Streamlit.

## ⚙️ Installation and Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/Employee-Attrition-KNN.git
cd Employee-Attrition-KNN
```

Replace `YOUR_GITHUB_USERNAME` with your GitHub username.

### 2️⃣ Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3️⃣ Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4
