COST_FUNCTION_NAMES = {
    "SigmoidCostFunction": "Sigmoid",
    "ExponentialCostFunction": "Exponential",
    "LogisticCostFunction": "Logistic",
    "SavageCostFunction": "Savage",
    "TangentCostFunction": "Tangent",
    "LogitAdjustedLogisticCostFunction": "Log. adj. logistic",
    "LogitAdjustedSavageCostFunction": "Log. adj. savage",
}
DATASET_NAMES = {
    "MNIST": "MNIST",
    "FashionMNIST": "F-MNIST",
    "KMNIST": "K-MNIST",
    "CIFAR10": "CIFAR-10",
}
METRIC_NAMES = {
    "positive_risk": "Positive risk",
    "negative_risk": "Negative risk",
    "erm_loss": "Empirical risk",
    "step_direction": "Step direction",
    "accuracy": "Accuracy",
    "f1_score": "F1 score",
    "balanced_accuracy": "Bal. accuracy",
    "auroc": "ROC AUC",
}
MIXUP_LOSS_NAMES = {"ChenMixupLoss": "Chen", "ZhaoMixupLoss": "Zhao"}
RISK_ESTIMATOR_NAMES = {
    "PNSystem": "PN",
    "uPUSystem": "uPU",
    "nnPUSystem": "nnPU",
    "ImbalancednnPUSystem": "Imba. nnPU",
}
SUBSET_NAMES = {"train": "Train", "val": "Test", "test": "Test"}
TARGET_NAMES = {"CIFAR10": {0: "airplane", 1: "automobile", 8: "ship", 9: "truck"}}
