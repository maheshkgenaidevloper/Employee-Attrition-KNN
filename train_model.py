
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "employee_attrition.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_FILE = MODEL_DIR / "knn_model.pkl"


def train_model():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    numeric_features = [
        "Age",
        "MonthlyIncome",
        "JobSatisfaction",
        "YearsAtCompany",
        "WorkLifeBalance"
    ]

    categorical_features = ["OverTime"]

    required_columns = (
        numeric_features + categorical_features + ["Attrition"]
    )

    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")

    X = df[numeric_features + categorical_features]
    y = df["Attrition"].astype(str).str.strip()

    if y.nunique() < 2 or y.value_counts().min() < 2:
        raise ValueError(
            "The dataset needs at least two classes, "
            "with at least two examples per class."
        )

    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler(), numeric_features),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", KNeighborsClassifier(n_neighbors=3))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\nKNN EMPLOYEE ATTRITION MODEL")
    print("-" * 35)
    print(f"Dataset records: {len(df)}")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.2%}")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    labels = sorted(y.unique())
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions, labels=labels))
    print("Class order:", labels)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_FILE)

    print(f"\nModel saved successfully: {MODEL_FILE}")


if __name__ == "__main__":
    train_model()
