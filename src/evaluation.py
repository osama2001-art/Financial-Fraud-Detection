
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)


def evaluate_model(y_test, y_pred):

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix:")
    print(cm)

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Accuracy
    print("\nAccuracy:")
    print(accuracy_score(y_test, y_pred))
