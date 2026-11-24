
from src.data_analysis import load_data, explore_data
from src.feature_engineering import create_features
from src.model_training import train_model
from src.evaluation import evaluate_model
from pathlib import Path


def main():

    # 1. Load dataset
    data = load_data()

    # 2. Explore dataset
    explore_data(data)

    # 3. Create features
    data = create_features(data)

    # 4. Train
    model, rf_model, lr_model, X_test, y_test, y_pred, rf_predictions, lr_predictions, lr_risk_scores = train_model(data)

    print("\n========== DECISION TREE RESULTS ==========")
    evaluate_model(y_test, y_pred)

    print("\n========== RANDOM FOREST RESULTS ==========")
    evaluate_model(y_test, rf_predictions)

    print("\n========== LOGISTIC REGRESSION RESULTS ==========")
    evaluate_model(y_test, lr_predictions)

    print("\n========== MODEL COMPARISON ==========")

    models = {
        "Decision Tree": y_pred,
        "Random Forest": rf_predictions,
        "Logistic Regression": lr_predictions
    }

    from sklearn.metrics import precision_score, recall_score, f1_score

    for name, predictions in models.items():
        print(f"\n{name}")
        print("Precision:", precision_score(y_test, predictions))
        print("Recall:", recall_score(y_test, predictions))
        print("F1 Score:", f1_score(y_test, predictions))

    # Classify transaction risk

    risk_levels = []

    for score in lr_risk_scores:
        if score >= 0.80:
            risk_levels.append("HIGH")
        elif score >= 0.40:
            risk_levels.append("MEDIUM")
        else:
            risk_levels.append("LOW")

    print("\n========== FRAUD RISK LEVELS ==========")

    import pandas as pd

    risk_summary = pd.Series(risk_levels).value_counts()

    print(risk_summary)

    # Evaluate high-risk alerts

    high_risk = lr_risk_scores >= 0.80

    high_risk_actual = y_test[high_risk]

    print("\n========== HIGH-RISK ALERT PERFORMANCE ==========")

    print("High-risk transactions:", high_risk.sum())
    print("Actual fraud cases:", high_risk_actual.sum())

    print(
        "Alert precision:",
        high_risk_actual.sum() / high_risk.sum()
    )

    print(
        "Fraud detection rate:",
        high_risk_actual.sum() / y_test.sum()
    )

    # Create fraud alert report

    alerts = X_test.copy()

    alerts["Actual_Fraud"] = y_test.values
    alerts["Fraud_Risk_Score"] = lr_risk_scores
    alerts["Risk_Level"] = risk_levels

    alerts = alerts[alerts["Risk_Level"] != "LOW"]

    alerts = alerts.sort_values(
        by="Fraud_Risk_Score",
        ascending=False
    )

    output_path = Path(__file__).resolve().parent / "fraud_alerts.csv"
    alerts.to_csv(output_path, index=False)

    print("Fraud alert report saved to:", output_path)

    print("\nFraud alert report saved successfully!")
    print(alerts.head(10))


if __name__ == "__main__":
    main()


