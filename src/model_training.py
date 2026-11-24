
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def train_model(data):
    """
    Prepare the data and train a Decision Tree model.
    """

    # Select relevant features
    features = [
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "balance_diff",
        "amount_balance_ratio"
    ]

    data = data.sort_values("step")

    # Get all unique time steps
    unique_steps = sorted(data["step"].unique())

    # Choose the 80% time boundary
    split_step_index = int(len(unique_steps) * 0.8)
    cutoff_step = unique_steps[split_step_index]

    # Split without sharing any time steps
    train_data = data[data["step"] < cutoff_step].copy()
    test_data = data[data["step"] >= cutoff_step].copy()

    print("\nTime cutoff:", cutoff_step)

    print(
        "Training time range:",
        train_data["step"].min(),
        "-",
        train_data["step"].max()
    )

    print(
        "Testing time range:",
        test_data["step"].min(),
        "-",
        test_data["step"].max()
    )

    # Sample only the training data
    train_data = train_data.groupby(
        "isFraud", group_keys=False
    ).sample(frac=0.05, random_state=42)

    print("\nTraining samples:", len(train_data))
    print("Testing samples:", len(test_data))

    # Sample 5% of the training data, preserving both classes
    train_data = train_data.groupby(
        "isFraud", group_keys=False
    ).sample(frac=0.05, random_state=42)

    # Separate features and target
    X_train = train_data[features]
    y_train = train_data["isFraud"]

    X_test = test_data[features]
    y_test = test_data["isFraud"]

    # Convert transaction types into numerical columns
    X_train = pd.get_dummies(X_train, columns=["type"], dtype=int)
    X_test = pd.get_dummies(X_test, columns=["type"], dtype=int)

    # Ensure both datasets have identical columns
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    print(
        "\nTraining time range:",
        train_data["step"].min(),
        "-",
        train_data["step"].max()
    )

    print(
        "Testing time range:",
        test_data["step"].min(),
        "-",
        test_data["step"].max()
    )

    # Create Decision Tree
    model = DecisionTreeClassifier(
        max_depth=8,
        class_weight="balanced",
        random_state=42
    )

    # Train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    # Logistic Regression
    lr_model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )

    lr_model.fit(X_train, y_train)
    lr_predictions = lr_model.predict(X_test)

    # Fraud risk scores
    lr_risk_scores = lr_model.predict_proba(X_test)[:, 1]

    print("\nFraud risk scores generated!")

    print("\nLogistic Regression training completed!")

    rf_model.fit(X_train, y_train)

    print("\nRandom Forest training completed!")

    # Make Random Forest predictions
    rf_predictions = rf_model.predict(X_test)

    # Train the model
    model.fit(X_train, y_train)


    print("\nDecision Tree training completed!")

    # Make predictions
    y_pred = model.predict(X_test)

    # Feature importance analysis

    print("\n========== DECISION TREE FEATURE IMPORTANCE ==========")

    dt_importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": model.feature_importances_
    })

    dt_importance = dt_importance.sort_values(
        by="Importance",
        ascending=False
    )

    print(dt_importance.to_string(index=False))

    print("\n========== RANDOM FOREST FEATURE IMPORTANCE ==========")

    rf_importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": rf_model.feature_importances_
    })

    rf_importance = rf_importance.sort_values(
        by="Importance",
        ascending=False
    )

    print(rf_importance.to_string(index=False))

    return model, rf_model, lr_model, X_test, y_test, y_pred, rf_predictions, lr_predictions, lr_risk_scores
