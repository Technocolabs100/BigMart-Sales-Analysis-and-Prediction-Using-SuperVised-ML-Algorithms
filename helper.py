import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import textwrap
from scipy.stats import mode
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder,StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder

NUM_COLUMNS = ["Item_Weight","Item_Visibility","Item_MRP","Item_Outlet_Sales","Outlet_Age"]  

CAT_COLUMNS = []

OHE_COLUNMS = ["Item_Type","Outlet_Type","Outlet_Location_Type","Item_Fat_Content","Outlet_Size","Outlet_Identifier"]

categorical_columns = ["Item_Fat_Content","Item_Type","Outlet_Location_Type","Outlet_Type","Outlet_Establishment_Year","Outlet_Size"]


def find_outliers_iqr(column):
    q1 = np.percentile(column, 25)
    q3 = np.percentile(column, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = np.where((column < lower_bound) | (column > upper_bound))[0]
    return outliers
    
def dist_plot(data):
    
    """
    Create a distribution plot for numerical data.

    Parameters:
    - data (array-like): Numeric data to be visualized.

    Returns:
    None
    """

    fig, axes = plt.subplots(2, 1, figsize=(10, 8),gridspec_kw={'height_ratios': [1, 2]})  
    plt.title("")
    mean_value = np.mean(data)
    
    median_value = np.nanmedian(data)
    
    mode_result = mode(data, nan_policy='omit')
    mode_value = mode_result.mode

    axes[1].set_title("")
    axes[0].set_title("")


    sns.histplot(data, bins=30, kde=True, color='blue', edgecolor='black',alpha=0.7, ax=axes[1]).set(xlabel=None)


    # Add vertical lines for mean, median, and mode on the first subplot
    axes[1].axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_value:.2f}')
    axes[1].axvline(median_value, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_value:.2f}')
    axes[1].axvline(mode_value, color='purple', linestyle='dashed', linewidth=2, label=f'Mode: {mode_value:.2f}')
    axes[1].legend()

    
    sns.boxplot(x=data, ax=axes[0], color='orange').set(xlabel=None)
    sns.stripplot(x=data, ax=axes[0], color="#474646").set(xlabel=None)
    fig.suptitle(f" The Distribution Of {data.name}", fontsize=16)
    plt.tight_layout()
    
    # print(f"{data.name}")
    # print(data.describe())
    


def wrap_labels(ax, width,break_long_words=False ,**kwargs):
    rot = kwargs.get("rot", 0) 
    """
    Wrap x-axis tick labels to fit within a specified width.

    Parameters:
    - ax (matplotlib.axes.Axes): The axis object to modify.
    - width (int): Maximum width for each label.
    - break_long_words (bool): Whether to break long words when wrapping.

    Returns:
    None
    """
    
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                      break_long_words=break_long_words))
    ax.set_xticklabels(labels, rotation=rot)


def subplot(num_columns, width, height, data, columns, fun, wrap=False, **kwargs):
    """
    Create subplots for numerical columns in a DataFrame using Seaborn.

    Parameters:
    - num_columns (int): Number of columns for the subplot grid.
    - width (float): Width of the overall figure.
    - height (float): Height of the overall figure.
    - data (pd.DataFrame): DataFrame containing the numerical columns.
    - columns (list): List of column names to plot.
    - fun (function): Custom function to apply to each subplot.
    - **kwargs: Additional keyword arguments to be passed to the custom function.

    Returns:
    None
    """
    num_rows = -(-len(columns) // num_columns)  # Ceil division

    _, axes = plt.subplots(num_rows, num_columns, figsize=(width, height))
    
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    if num_columns == 1:
        axes = axes.reshape(-1, 1)
    
    for i, ax in enumerate(axes.flat):
        if i < len(columns):
            column_name = columns[i]

            fun(data , column_name, ax, **kwargs)
            if wrap == True :
                wrap_labels(ax, 10,**kwargs)
                ax.figure


    plt.tight_layout()
    plt.show()

def encoder(num, cat, ohe, pca_num_components=None):
    transformers = [
        ('num', StandardScaler(), num),
        ('cat', OrdinalEncoder(), cat),
        ('OHE', OneHotEncoder(), ohe)
    ]

    # Add PCA if specified
    if pca_num_components is not None:
        transformers.append(('PCA', PCA(n_components=pca_num_components), num))

    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor

def preprocess_data(data, target_column):
    categorical_columns_copy = CAT_COLUMNS.copy()
    numerical_columns_copy = NUM_COLUMNS.copy()

    if target_column in categorical_columns_copy:
        categorical_columns_copy.remove(target_column)
    else:
        numerical_columns_copy.remove(target_column)

    X = data.drop([target_column], axis=1)
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = encoder(numerical_columns_copy,categorical_columns_copy,OHE_COLUNMS)
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    return X_train_preprocessed, X_test_preprocessed, y_train, y_test, preprocessor


def preprocess_data_for_nulls(df, target_column, sec = 0 ):
    """
    Preprocesses the input DataFrame for machine learning, preparing it for training and prediction.

    Parameters:
    - df: pd.DataFrame
        The input DataFrame containing features and the target column.
    - target_column: str
        The name of the target column to be predicted.
    - numerical_columns_copy: list or None, optional
        List of numerical columns. If None, it defaults to all numerical columns in the DataFrame.

    Returns:
    - tuple
        A tuple containing the preprocessor, training features (X_train), training labels (y_train),
        and a DataFrame containing instances with missing values in the target column (df_predict).
    
    Notes:
    - The training data consists of instances where the target column is not null, and prediction data consists
      of instances where the target column is null.
    - Categorical columns are specified in "categorical_columns", and numerical columns in "numerical_columns".
    - The preprocessor is a sklearn ColumnTransformer that scales numerical columns using StandardScaler
      and encodes categorical columns using OrdinalEncoder.
    """
    copyy = NUM_COLUMNS.copy()

    if sec == 0 :
        copyy.remove("Item_Outlet_Sales")

    df_train = df[df[target_column].notnull()]
    df_predict = df[df[target_column].isnull()]

    X_train = df_train.drop([target_column], axis=1)
    y_train = df_train[target_column]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), copyy),
            ('cat', OrdinalEncoder(), categorical_columns[:-2]),
        ])

    return preprocessor, X_train, y_train, df_predict


def load_model(preprocessor, model, X_train, y_train, X_test, y_test, metric):
    """
    Train a machine learning model using a preprocessor and evaluate its performance.

    Parameters:
    - preprocessor: sklearn transformer
        The data preprocessing steps to be applied before training the model.
    - model: sklearn estimator
        The machine learning model to be trained and evaluated.
    - X_train: array-like or pd.DataFrame
        The input features for training the model.
    - y_train: array-like or pd.Series
        The target labels for training the model.
    - X_test: array-like or pd.DataFrame
        The input features for evaluating the model.
    - y_test: array-like or pd.Series
        The target labels for evaluating the model.
    - metric: callable
        The performance metric function used for evaluation.

    Returns:
    - tuple
        A tuple containing the training and testing performance scores based on the specified metric.
    """
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    pipeline.fit(X_train, y_train)

    return metric(y_train, pipeline.predict(X_train)), metric(y_test, pipeline.predict(X_test))


def evaluate_classifier(model_name, preprocessor, model, X_train, y_train, X_test, y_test,):
    """
    Train and evaluate a classification model.

    Parameters:
    - model_name: str
        The name or identifier for the model.
    - preprocessor: sklearn transformer
        The data preprocessing steps to be applied before training the model.
    - model: sklearn classifier
        The classification model to be trained and evaluated.
    - X_train: array-like or pd.DataFrame
        The input features for training the model.
    - y_train: array-like or pd.Series
        The target labels for training the model.
    - X_test: array-like or pd.DataFrame
        The input features for evaluating the model.
    - y_test: array-like or pd.Series
        The target labels for evaluating the model.
    - metric: callable
        The performance metric function used for evaluation.

    Prints:
    - str
        The training and testing accuracy of the classifier.
    """
    train_accuracy, test_accuracy = load_model(preprocessor, model, X_train, y_train, X_test, y_test, accuracy_score)
    
    print(f'{model_name} - Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')


def evaluate_model(model_name, preprocessor, model, X_train, y_train, X_test, y_test):
    """
    Train and evaluate a regression model.

    Parameters:
    - model_name: str
        The name or identifier for the model.
    - preprocessor: sklearn transformer
        The data preprocessing steps to be applied before training the model.
    - model: sklearn regressor
        The regression model to be trained and evaluated.
    - X_train: array-like or pd.DataFrame
        The input features for training the model.
    - y_train: array-like or pd.Series
        The target labels for training the model.
    - X_test: array-like or pd.DataFrame
        The input features for evaluating the model.
    - y_test: array-like or pd.Series
        The target labels for evaluating the model.
    - metric: callable
        The performance metric function used for evaluation.

    Prints:
    - str
        The training and testing mean squared error (MSE) of the regression model.
        Additionally, provides insights into potential overfitting or underfitting.
    """
    train_mse, test_mse = load_model(preprocessor, model, X_train, y_train, X_test, y_test, mean_squared_error)
    
    print(f'{model_name} - Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}')

    if train_mse < test_mse:
        print(f'The model may be overfitting as Train MSE ({train_mse:.4f}) < Test MSE ({test_mse:.4f})')
    elif train_mse > test_mse:
        print(f'The model may be underfitting as Train MSE ({train_mse:.4f}) > Test MSE ({test_mse:.4f})')
    else:
        print('The model seems to generalize well.')

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Create a function to plot the results

def plot_results(results, X_test_preprocessed, y_test):
    
    for model_name, result in results.items():
        print(f"Model: {model_name}")
        print(f"Training MSE: {result['train_mse']}")
        print(f"Training RMSE: {result['train_rmse']}")
        print(f"Training R^2: {result['train_r2']}")
        print("="*50)
        print(f"Testing MSE: {result['test_mse']}")
        print(f"Testing RMSE: {result['test_rmse']}")
        print(f"Testing R^2: {result['test_r2']}")
        print("="*50)


