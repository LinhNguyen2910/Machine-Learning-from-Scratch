## Learning Approach

The Machine Learning algorithms in this project are implemented from scratch using Python and NumPy.

Scikit-learn is used only for data preprocessing and dataset preparation, including:

- Loading the Iris dataset
- Splitting data into training and testing sets
- Feature standardization using `StandardScaler`

The Machine Learning models themselves are implemented manually without using scikit-learn's built-in model implementations.

For example:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler