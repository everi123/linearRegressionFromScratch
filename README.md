# Linear Regression Mini Project

This mini project focuses on implementing a simple linear regression model from scratch and structuring the project to make it more maintainable. It also includes unit tests to ensure the correctness of the implementation.

## Project Structure

- `data/`: Directory for storing dataset(s).
  - `salary_data.csv`: Dataset containing salary information.
  
- `models/`: Directory for model-related code.
  - `__init__.py`: Initialization file for the models package.
  - `linear_regression.py`: Implementation of the `LinearRegressionModel` class.

- `utils/`: Directory for utility functions.
  - `__init__.py`: Initialization file for the utils package.
  - `data_preprocessing.py`: Data preprocessing functions.

- `tests/`: Directory for unit tests.
  - `__init__.py`: Initialization file for the tests package.
  - `test_linear_regression.py`: Unit tests for the `LinearRegressionModel`.

- `main.py`: Main script to run the project.

## Linear Regression Implementation

The linear regression model is implemented in the `LinearRegressionModel` class in `models/linear_regression.py`. The class includes methods for fitting the model to the training data (`fit`), making predictions (`predict`), and updating weights using gradient descent (`update_weights`).

## Data Preprocessing

Data preprocessing, including loading and splitting the dataset, is done in the `utils/data_preprocessing.py` module. The `load_and_preprocess_data` function reads the dataset from a CSV file and splits it into training and test sets using `train_test_split` from `sklearn`.

## Unit Tests

Unit tests for the `LinearRegressionModel` class are defined in `tests/test_linear_regression.py`. The tests cover the `fit` and `predict` methods, ensuring that the model behaves as expected.
## Conclusion

Through this mini project, I have learned how to structure a machine learning project, make it more maintainable, and use unit tests to validate the implementation. I have gained a deeper understanding of linear regression and the importance of testing in machine learning development. These skills are essential for building robust and reliable machine learning systems.




```bash 
# root dictory e.g: cd RegressionProject
# for testing the project
python -m unittest tests.test_linear_regression


## Running the Project 
python main.py 

