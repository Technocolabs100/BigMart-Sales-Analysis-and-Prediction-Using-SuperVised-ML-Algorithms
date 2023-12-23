import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import textwrap
from scipy.stats import mode
from sklearn.metrics import r2_score
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder,StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from itertools import chain



NUM_COLUMNS = ["Item_Weight","Item_Visibility","Item_MRP","Item_Outlet_Sales","Outlet_Age"]  

CAT_COLUMNS = []

OHE_COLUNMS = ["Item_Type","Outlet_Type","Outlet_Location_Type","Item_Fat_Content","Outlet_Size","Outlet_Identifier"]

categorical_columns = ["Item_Fat_Content","Item_Type","Outlet_Location_Type","Outlet_Type","Outlet_Establishment_Year","Outlet_Size"]


def find_outliers_iqr(column):
    """
    Detect outliers in a numerical column using the Interquartile Range (IQR) method.

    Parameters:
    - column: A NumPy array or Pandas Series representing the numerical column for which outliers are to be detected.

    Returns:
    - A NumPy array containing the indices of the outliers in the provided column.
    """
    q1 = np.percentile(column, 25)
    q3 = np.percentile(column, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = np.where((column < lower_bound) | (column > upper_bound))[0]
    
    return outliers

def outliers_detection_percent(columns, df):
    """
    Apply the find_outliers_iqr function to multiple numerical columns in a DataFrame
    and provide information about the percentage of outliers for each column.

    Parameters:
    - columns: A list of column names representing the numerical columns in the DataFrame.
    - df: A Pandas DataFrame containing the data.

    Returns:
    - A dictionary where keys are column names and values are arrays containing the indices of outliers in each corresponding column.
    - The function also prints the percentage of outliers in the entire DataFrame.
    """
    outliers = dict()

    for column_name in columns:
        outlier_indices = find_outliers_iqr(df[column_name])

        if len(outlier_indices) > 0:
            outliers.update({column_name: outlier_indices})

    indexes = set(chain(*outliers.values()))

    print(len(indexes) / df.shape[0])

    return outliers , set(chain(*outliers.values()))

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


    axes[1].axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_value:.2f}')
    axes[1].axvline(median_value, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_value:.2f}')
    axes[1].axvline(mode_value, color='purple', linestyle='dashed', linewidth=2, label=f'Mode: {mode_value:.2f}')
    axes[1].legend()

    
    sns.boxplot(x=data, ax=axes[0], color='orange').set(xlabel=None)
    sns.stripplot(x=data, ax=axes[0], color="#474646").set(xlabel=None)
    fig.suptitle(f" The Distribution Of {data.name}", fontsize=16)
    plt.tight_layout()
    



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


def select_optimal_components(data,desired_explained_variance=0.95):
    """
    Performs Principal Component Analysis (PCA) on a given dataset to determine
    the optimal number of components based on the desired explained variance.

    Parameters:
    - data: pandas DataFrame or numpy array, input dataset for PCA.
    - desired_explained_variance: float, optional, the desired cumulative explained variance (default is 0.95).

    Returns:
    - optimal_num_components: int, the optimal number of principal components.

    """
    
    X_std = StandardScaler().fit_transform(data)
    pca = PCA()
    X_pca = pca.fit_transform(X_std)

    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

    plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance vs. Number of Components')
    plt.show()

    optimal_num_components = int (np.argmax(cumulative_variance_ratio >= desired_explained_variance)) + 1

    print(f"Optimal number of components: {optimal_num_components}")

    return optimal_num_components


def encoder(data , num, cat, ohe,pca=False ,pca_num_components=None):
    """
    Creates a preprocessing pipeline using ColumnTransformer from scikit-learn.
    Includes standard scaling for numerical features, ordinal encoding for categorical
    variables, one-hot encoding, and optional PCA for numerical features.

    Parameters:
    - num: list
        Numerical feature column names.
    - cat: list
        Categorical feature column names.
    - ohe: list
        One-hot encoded column names.
    - pca: bool, optional
        Whether to include PCA for numerical features (default is False).
    - pca_num_components: int, optional
        The number of components for PCA. If not provided, it will be determined
        using the select_optimal_components function.

    Returns:
    - preprocessor: ColumnTransformer
        The preprocessing pipeline.
    """
        
    transformers = [
        ('num', StandardScaler(), num),
        ('cat', OrdinalEncoder(), cat),
        ('OHE', OneHotEncoder(), ohe)
    ]

    if pca:
        optimal_num_components = select_optimal_components(data[num])
        pca_num_components = pca_num_components if pca_num_components is not None else optimal_num_components
        transformers.append(('PCA', PCA(n_components=pca_num_components), num))

    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor


def preprocess_data(data, target_column, test_size=0.2, pca_num_components=None , pca=False ):
    """
    Preprocesses the input DataFrame for machine learning, preparing it for training and prediction.

    Parameters:
    - data: pd.DataFrame
        The input DataFrame containing features and the target column.
    - target_column: str
        The name of the target column to be predicted.
    - test_size: float, optional
        The proportion of the dataset to include in the test split (default is 0.2).
    - pca: bool, optional
        Whether to include PCA for numerical features (default is False).
    - pca_num_components: int, optional
        The number of components for PCA. If not provided, it will be determined
        using the select_optimal_components function.

    Returns:
    - tuple
        A tuple containing the preprocessor, training features (X_train),
        training labels (y_train), and a DataFrame containing instances with
        missing values in the target column (df_predict).

    Notes:
    - The training data consists of instances where the target column is not null,
      and prediction data consists of instances where the target column is null.
    - Categorical columns are specified in "categorical_columns", and numerical
      columns in "numerical_columns".
    - The preprocessor is a sklearn ColumnTransformer that scales numerical columns
      using StandardScaler and encodes categorical columns using OrdinalEncoder.
    """


    numerical_columns_copy = NUM_COLUMNS.copy()
    categorical_columns_copy = CAT_COLUMNS.copy()

    if target_column in categorical_columns_copy:
        categorical_columns_copy.remove(target_column)
    else:
        numerical_columns_copy.remove(target_column)

    X = data.drop([target_column], axis=1)
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    preprocessor = encoder(data,numerical_columns_copy, categorical_columns_copy, OHE_COLUNMS, pca=pca, pca_num_components=pca_num_components)

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


def null_columns (df):
    """
        Train a machine learning model using a preprocessor and evaluate its performance.

        Parameters:
        - df (pandas.DataFrame): The testing dataset.

        Returns:
        None

        Prints:
        - Percentage of null values for each column in the  dataset.
    """
    
    print("nulls: ")
    [print(f"{i}:{(v / df.shape[0]):.2%}") for i, v in df.isna().sum().items() if v != 0]



def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    
    # Predictions on training set
    y_train_pred = model.predict(X_train)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score(y_train, y_train_pred)
    
    # Predictions on testing set
    y_test_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_test_pred)
    
    results = {
        'model': model,
        'train_mse': train_mse,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'test_mse': test_mse,
        'test_rmse': test_rmse,
        'test_r2': test_r2
    }
    
    return results


def plot_rmse_comparison(results):
    rmse_values = {'Model': [], 'Train RMSE': [], 'Test RMSE': []}

    for model_name, result in results.items():
        rmse_values['Model'].append(model_name)
        rmse_values['Train RMSE'].append(result['train_rmse'])
        rmse_values['Test RMSE'].append(result['test_rmse'])

    rmse_df = pd.DataFrame(rmse_values)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='value', hue='variable', data=pd.melt(rmse_df, ['Model']))
    plt.title('Comparison of RMSE on Training and Testing Sets')
    plt.ylabel('Root Mean Squared Error (RMSE)')
    plt.show()