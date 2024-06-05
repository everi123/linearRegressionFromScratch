from Regression.models.Regression import LinearRegressionModel
from utils.data_preprocessing import load_and_preprocess_data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
def main():
    # Load and preprocess data
    X_train, X_test, Y_train, Y_test = load_and_preprocess_data("data/salary_data.csv")

    # Model training
    model = LinearRegressionModel()
    model.fit(X_train, Y_train)

    # Prediction
    Y_pred = model.predict(X_test)
    print("Predicted values ", np.round(Y_pred[:3], 2))
    print("Real values      ", Y_test[:3])
    print("Trained W        ", round(model.W[0], 2))
    print("Trained b        ", round(model.b, 2))

    # Visualization on test set
    plt.scatter(X_test, Y_test, color='blue')
    plt.plot(X_test, Y_pred, color='orange')
    plt.title('Salary vs Experience')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.show()

if __name__ == "__main__":
    main()