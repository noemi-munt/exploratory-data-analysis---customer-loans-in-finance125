import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import seaborn as sns

class DataTransform:
    """
    A class to perform various data transformation operations on a DataFrame.

    Methods:
    convert_to_category(column_names: list):
        Convert specified columns to category type.
        
    convert_to_ordered_category(ordered_cat_dict: dict):
        Convert specified columns to ordered category type.
        
    convert_to_date(column_names: list):
        Convert specified columns to datetime type.
    """
    def __init__(self, df: pd.DataFrame):
        """
        Initialise the DataTransform object with a DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame to be transformed.
        """
        self.df = df

    def convert_to_category(self, column_names: list):
        """
        Convert specified columns to category type.

        Parameters:
        column_names (list): List of column names to convert to category type.

        Returns:
        pd.DataFrame: The DataFrame with specified columns converted to category type.
        """
        for col in column_names:
            self.df[col] = self.df[col].astype('category')
        return self.df

    def convert_to_ordered_category(self, ordered_cat_dict: dict):
        """
        Convert specified columns to ordered category type.

        Parameters:
        ordered_cat_dict (dict): Dictionary with column names as keys and lists with the ordered values of each category.

        Returns:
        pd.DataFrame: The DataFrame with specified columns converted to ordered category type.
        """
        #loop through the dicitonary's keys and values
        for col, order in ordered_cat_dict.items():
            cat_type = CategoricalDtype(categories=order, ordered=True)
            self.df[col] = self.df[col].astype(cat_type)
        return self.df

    def convert_to_date(self, date_columns: list):
        """
        Convert specified columns to datetime type.

        Parameters:
        date_columns (list): List of column names to convert to datetime type.

        Returns:
        pd.DataFrame: The DataFrame with specified columns converted to datetime type.
        """
        for col in date_columns:
            self.df[col] = pd.to_datetime(self.df[col], format='mixed')
        return self.df


class DataFrameInfo:
    """
    A class to provide information about a DataFrame.
    
    Methods:
    describe_columns: Describe all columns in the DataFrame to check their data types.
    extract_statistics: Extract statistical values: median, standard deviation, and mean from the columns of the DataFrame.
    count_distinct_values: Count distinct values in categorical columns.
    print_shape: Print out the shape of the DataFrame.
    count_null_values: Generate a count & percentage count of NULL values in each column.
    """
    def __init__(self, df: pd.DataFrame):
        """
        Initialise the DataFrameInfo object with a DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame to be analysed.
        """
        self.df = df

    def describe_columns(self):
        """Describe all columns in the DataFrame to check their data types."""
        return self.df.dtypes

    def extract_statistics(self):
        """Extract statistical values: median, standard deviation, and mean from the columns of the DataFrame."""
        statistics = {
            'Mean': self.df.mean(numeric_only=True),
            'Median': self.df.median(numeric_only=True),
            'Std Dev': self.df.std(numeric_only=True)
        }
        return statistics

    def count_distinct_values_and_mode(self):
        """
        Count distinct values in categorical columns and compute the mode (most frequent value) for each.
        
        Returns:
            dict: A dictionary where each column maps to a dictionary with 'distinct_count' and 'mode'.
        """
        # Select categorical columns
        categorical_columns = self.df.select_dtypes(include=['category', 'object']).columns

        # Compute distinct counts and mode
        stats = {}
        for col in categorical_columns:
            mode_values = self.df[col].mode()  # Get the mode(s)
            stats[col] = {
                'Distinct Count': self.df[col].nunique(),
                'Mode': mode_values.iloc[0] if not mode_values.empty else None  # Handle empty columns
            }
    
        return stats
    
    def print_shape(self):
        """Print out the shape of the DataFrame."""
        return self.df.shape

    def count_null_values(self):
        """Generate a count & percentage of NULL values in each column."""
        null_counts = self.df.isnull().sum()
        null_percentages = (null_counts / len(self.df)) * 100
        null_summary = pd.DataFrame({'count': null_counts, 'percentage': null_percentages})
        return null_summary


class DataFrameTransform:
    """
    A class to perform transformations on a DataFrame.
    """
    def __init__(self, df: pd.DataFrame):
        """
        Initialise the DataFrameTransform object with a DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame to be transformed.
        """
        self.df = df
    
    def drop_columns(self, null_summary: pd.DataFrame, threshold: float):
        """
        Drop columns with NULL value percentages greater than a specified threshold.

        Parameters:
        null_summary (pd.DataFrame): DataFrame containing count and percentage of NULL values in each column.
        threshold (float): The percentage threshold above which columns will be dropped.
        """
        filtered_df = null_summary[null_summary['percentage'] > threshold]
        id_list = filtered_df.index.to_list()
        print(f"Columns to drop: {id_list}")

        self.df.drop(id_list, axis=1, inplace=True)
        print(f"New shape: {self.df.shape}")
    
    def drop_rows(self, null_summary: pd.DataFrame, threshold: float):
        """
        Drop rows with NULL value percentages less than a specified threshold.

        Parameters:
        null_summary (pd.DataFrame): DataFrame containing count and percentage of NULL values in each column.
        threshold (float): The percentage threshold below which rows will be considered for dropping.
        """
        filtered_df = null_summary[null_summary['percentage'] < threshold]
        filtered_df = filtered_df[filtered_df['percentage'] > 0] #remove columns with 0% null count
        id_list = filtered_df.index.to_list()
        print(f"Drop rows with null values from: {id_list}")

        self.df.dropna(axis=0, subset=id_list, inplace=True)
        print(f"New shape: {self.df.shape}")
       
    
    def impute_median(self, col_name: str):
        """
        Impute NULL values of a column with its median.

        Parameters:
        col_name (str): The column name for which NULL values are to be imputed.
        """
        return self.df[col_name].fillna(self.df[col_name].median())
    
    def skew_transform(self, subset_columns: list):
        """
        Perform transformations on columns to determine which transformation results in the biggest reduction in skew.

        Parameters:
        subset_columns (list): List of column names to be transformed.

        Returns:
        pd.DataFrame: The DataFrame with transformed columns.
        """
        def log_transform(series: pd.Series):
            return np.log1p(series)

        def sqrt_transform(series: pd.Series):
            return np.sqrt(series)

        def reciprocal_transform(series: pd.Series):
            return 1 / (series + 1e-9)

        transformation_results = {}
        best_transformations_dict = {}  

        for col in subset_columns:
            original_skew = self.df[col].skew()
            log_skew = log_transform(self.df[col]).skew()
            sqrt_skew = sqrt_transform(self.df[col]).skew()
            reciprocal_skew = reciprocal_transform(self.df[col]).skew()

            transformation_results[col] = {
                'Original': original_skew,
                'Log': log_skew,
                'Square Root': sqrt_skew,
                'Reciprocal': reciprocal_skew,
            }

            # Find the best transformation (minimum absolute skew)
            best_transformation = min(
                ['Log', 'Square Root', 'Reciprocal'], 
                key=lambda t: abs(transformation_results[col][t])
            )

            best_transformations_dict[col] = best_transformation  


        skew_df = pd.DataFrame(transformation_results).T # Transpose df
        print(skew_df)

        transformed_df = self.df.copy()
        
        for col, best_transformation in best_transformations_dict.items():
            if best_transformation == 'Log':
                transformed_df[col] = log_transform(self.df[col])
            elif best_transformation == 'Square Root':
                transformed_df[col] = sqrt_transform(self.df[col])
            elif best_transformation == 'Reciprocal':
                transformed_df[col] = reciprocal_transform(self.df[col])

        return transformed_df, best_transformations_dict

    def plot_skew_transformations(self, subset_columns: list, best_transformations_dict: dict):
        """
        Plots histograms for each column in the subset, showing original and transformed distributions,
        highlighting the best transformation in a different color.
        
        Parameters:
        subset_columns (list): List of column names to be visualized.
        skew_df (pd.DataFrame): DataFrame containing skew information and best transformation.
        """
        num_columns = len(subset_columns)
        fig, axes = plt.subplots(num_columns, 4, figsize=(16, 4 * num_columns))


        transformations = {
            "Original": lambda x: x,
            "Square Root": np.sqrt,
            "Log": np.log1p,
            "Reciprocal": lambda x: 1 / (x + 1e-9)
        }

        default_color = "blue"
        highlight_color = "tomato"

        for i, col in enumerate(subset_columns):
            best_transformation = best_transformations_dict[col]
            for j, (title, transform) in enumerate(transformations.items()):
                ax = axes[i, j]
                transformed_data = transform(self.df[col])  # apply transformation to column
                
                color = highlight_color if title == best_transformation else default_color
                
                sns.histplot(transformed_data, ax=ax, kde=True, bins=30, color=color)
                ax.set_title(f"{col} - {title}")

        plt.tight_layout()
        plt.show()
    
    def remove_outliers_iqr(self, column):
        """
        Calculate the interquartile range of a column to remove outliers.

        Parameters:
        column (str): Name of column to remove outliers.

        Returns:
        pd.DataFrame: The DataFrame with transformed columns.
        """
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_threshold = Q1 - 1.5 * IQR
        upper_threshold = Q3 + 1.5 * IQR
        
        # Remove outliers based on thresholds
        df_filtered = self.df[(self.df[column] >= lower_threshold) & (self.df[column] <= upper_threshold)]
        return df_filtered
    
    def remove_outliers_modified_iqr(self, column):
        """
        Removes outliers from a column using a modified IQR method that accounts for many zero values.
        
        Parameters:
        column (str): Column to remove outliers from.
        
        Returns:
        pd.DataFrame: Filtered DataFrame with nonzero values preserved.
        """
        nonzero_data = self.df[self.df[column] > 0][column]  # Consider only nonzero values for IQR method
        
        Q1 = nonzero_data.quantile(0.25)
        Q3 = nonzero_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_threshold = Q1 - 3 * IQR 
        upper_threshold = Q3 + 3 * IQR

        # Keep all zero values, and apply threshold filtering only to nonzero values
        df_filtered = self.df[
            (self.df[column] == 0) | ((self.df[column] >= lower_threshold) & (self.df[column] <= upper_threshold))
        ]
        
        return df_filtered


class Plotter:
    """
    A class to visualize insights from the data.
    """
    def __init__(self, df):
        """
        Initialise the Plotter object with data to plot.

        Parameters:
        df (pd.DataFrame): The DataFrame to be plotted.
        """
        self.df = df
    
    def boxplot(self, x_label: str, title: str):
        """
        Plot a boxplot of the data.

        Parameters:
        xlabel (str): The label for the x-axis.
        title (str): The title of the plot.        
        """
        sns.boxplot(y=x_label, data=self.df)
        plt.title(f'Box Plot of {title}')
        plt.show()
    

    def facetgrid(self, cols:list, plot):
        """
        Create multi-grid plots of the data.

        Parameters:
        cols (list): The list of column to be plotted.
        plot (str): The type of plot.     
        """
        
        f = pd.melt(self.df, value_vars=cols)
        g = sns.FacetGrid(f, col="variable",  col_wrap=3, sharex=False, sharey=False)
        if plot == "box":
            g = g.map(sns.boxplot, "value")
        else:
            g = g.map(sns.histplot, "value", kde=True)

    def heatmap(self, data, title):
        """
        Plot a heat map of the data.

        Parameters:
        data (pd.DataFrame): The DataFrame to be mapped.
        Title (str): Title of the plot.     
        """
        plt.figure(figsize=(15, 8))
        sns.heatmap(data, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title(title)
        plt.show()
    
    def histogram(self, x_val, x_label: str, title: str):
        """
        Plot a histogram of the data.

        Parameters:
        xlabel (str): The label for the x-axis.
        title (str): The title of the plot.
        """
        sns.histplot(x_val, bins=10, kde=True)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.xlabel(x_label)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.title(title)
        plt.show()

    def countplot(self, plot_data, x_col, y_col, data_order, title, xlabel, ylabel):
        """
        Plot bar charts of categorical data according to Loan Status.

        Parameters:
        data (pd.DataFrame): The DataFrame to be mapped.
        Title (str): Title of the plot.     
        """
        
        sns.countplot(data=plot_data, x=x_col, y=y_col, hue='loan_status', hue_order=["Fully Paid","Late","Charged Off"], palette='coolwarm',order=data_order)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(title="Risk Status")
        plt.show()
    
