import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# -----------------------------------------
# 1. Load dataset
# -----------------------------------------

data = pd.read_csv("student_placement_dataset.csv")

print("Dataset loaded successfully!")
print(data.head())


# -----------------------------------------
# 2. Separate features and target
# -----------------------------------------

X = data.drop(["Placement", "Student_ID"], axis=1)

y = data["Placement"]


# -----------------------------------------
# 3. Split into training and testing data
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=15,
    stratify=y
)


# -----------------------------------------
# 4. Try different tree depths
# -----------------------------------------

depths = [3, 4, 5]

results = []


for depth in depths:

    # Create model
    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=15
    )

    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    # Store results
    results.append([
        depth,
        accuracy,
        precision,
        recall,
        f1
    ])


# -----------------------------------------
# 5. Comparative analysis
# -----------------------------------------

results_df = pd.DataFrame(
    results,
    columns=[
        "Max Depth",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)


print("\nDecision Tree Comparison:")
print(results_df.to_string(index=False))