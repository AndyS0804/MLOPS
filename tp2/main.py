import mlflow
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main():
    # Use a local directory for MLflow tracking to avoid permission issues
    mlflow.set_tracking_uri("file:./mlflow/mlruns")

    X, y = datasets.load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    params = {
        "solver": "lbfgs",
        "max_iter": 100,
        "random_state": 42,
    }

    mlflow.set_experiment("iris_classifier")

    with mlflow.start_run():
        mlflow.log_params(params)

        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("dataset", "iris")

        lr = LogisticRegression(**params)
        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(lr, "iris_classifier")

        mlflow.set_tag("Training Info", "Logistic Regression for Iris classification")

        run_id = mlflow.active_run().info.run_id
        print(f"Run ID: {run_id}")


if __name__ == "__main__":
    main()
