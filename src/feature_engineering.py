def create_features(data):
    """
    Create additional features for fraud detection.
    """

    # Difference between sender's old and new balance
    data["balance_diff"] = (
        data["oldbalanceOrg"] - data["newbalanceOrig"]
    )

    # Transaction amount relative to sender's balance
    data["amount_balance_ratio"] = (
        data["amount"] / (data["oldbalanceOrg"] + 1)
    )

    print("\nFeature engineering completed!")

    return data