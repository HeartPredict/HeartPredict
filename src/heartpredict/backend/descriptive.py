"""Utilities for conducting a descriptive data analysis"""
import matplotlib.pyplot as plt
import pandas as pd
import logging
from heartpredict.backend.data import ProjectData
from dataclasses import dataclass

MEANING_BINARY_COLUMNS = {
    "anaemia": {0: "No anaemia", 1: "anaemia"},
    "diabetes": {0: "No diabetes", 1: "diabetes"},
    "high_blood_pressure": {0: "Normal blood pressure", 1: "High blood pressure"},
    "sex": {0: "Female", 1: "Male"},
    "smoking": {0: "Not smoking", 1: "Is smoking"},
    "DEATH_EVENT": {0: "Survived", 1: "Died"},
}

@dataclass
class DiscreteStatistics:
    name: str
    minimum: float
    maximum: float
    median: float
    mean: float
    standard_dev: float

@dataclass
class BooleanStatistics:
    name: str
    zero: float
    one: float
    

class DataFrameAnalyzer:
    def __init__(self, 
                 path_to_data: str = "data/heart_failure_clinical_records.csv") -> None:
        logging.debug("Instantiate ProjectData object")
        project_data = ProjectData(path_to_data)
        self.df = project_data.df
        logging.debug("DataFrame added to DataFrameAnalyzer")

    def calculate_boolean_statistics(self, boolean_column: str) -> BooleanStatistics:
        """
        Create a BooleanStatistics object containing main statistics

        Args:
            boolean_column: Boolean DataFrame column (e.g. smoking)

        Returns:
            BooleanStatistics object
        """
        logging.debug("Read in Boolean column")
        col_data = self.df[boolean_column]
        col_size = len(col_data)
        col_distribution = col_data.value_counts().to_dict()
        zero_val = col_distribution[0] / col_size
        one_val = col_distribution[1] / col_size
        logging.debug("Boolean statistics calculated")

        return BooleanStatistics(name=boolean_column,
                                 zero=zero_val,
                                 one=one_val)


    def calculate_discrete_statistics(self, discrete_column: str) -> DiscreteStatistics:
        """
        Create a DiscreteStatistics object containing main statistics

        Args:
            discrete_column: Discrete DataFrame column (e.g. age)

        Returns:
            DiscreteStatistics object
        """
        logging.debug("Read in Discrete column")
        col_data = self.df[discrete_column]
        min_val = col_data.min()
        max_val = col_data.max()
        median_val = col_data.median()
        mean_val = col_data.mean()
        standard_dev_val = col_data.std()
        logging.debug("Discrete statistics calculated")

        return DiscreteStatistics(name=discrete_column,
                                  minimum=min_val,
                                  maximum=max_val,
                                  median=median_val,
                                  mean=mean_val,
                                  standard_dev=standard_dev_val)
    

    def create_conditional_dataset(self, 
                                   col: str, 
                                   num: int, 
                                   rel: str,
                                   df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Create a filtered dataset, e.g. only patients over 60

        Args:
            df: DataFrame of (sub)dataset
            col: Column name for filtering (e.g. 'age')
            num: Number for filtering (e.g. 60)
            rel: Relation for filtering (e.g. ">")

        Returns:
            DataFrame of filtered dataset
        """
        # Check which DataFrame to use
        logging.debug("Read-in DataFrame")
        if df is None:
            df = self.df
        
        # Check the condition's relation
        logging.debug("Assemble filter condition")
        if rel == "==":
            cond = df[col] == num
        elif rel == "<":
            cond = df[col] < num
        elif rel == ">":
            cond = df[col] > num
        elif rel == "<=":
            cond = df[col] <= num
        elif rel == ">=":
            cond = df[col] >= num

        # Return conditioned dataset
        logging.debug("Create conditional DataFrame")
        df_cond = df[cond].copy()
        return df_cond
    

    def save_variable_distribution(self, 
                                   column: str,
                                   df: pd.DataFrame = None) -> dict:
        """
        Save unique variable expressions of a DataFrame column in a dict

        Args:
            df: DataFrame of (sub)dataset
            col: Column to be analyzed

        Returns:
            Dictionary counting the variable expressions
        """
        # Check which DataFrame to use
        logging.debug("Read-in DataFrame")
        if df is None:
            df = self.df
        
        # Check if the column is boolean or discrete
        logging.debug("Check if column is boolean/discrete")
        condition = set(df[column].unique()) == {0,1}
        if condition:
            logging.debug("Interpret Boolean distribution")
            interpreted_distribution = {}
            bool_meaning = MEANING_BINARY_COLUMNS[column]
            distribution = df[column].value_counts().to_dict()
            for num in distribution.keys():
                interpreted_distribution[bool_meaning[num]] = distribution[num]
            logging.debug("Return Boolean distribution")
            return interpreted_distribution

        else:
            logging.debug("Create distribution dict for Discrete column")
            distribution = df[column].value_counts().to_dict()
            logging.debug("Return Discrete distribution")
            return distribution


def save_distribution_plot(distribution: dict, col_name: str) -> tuple:
    """
    Create and return a simple bar plot for a specific column

    Args:
        distribution: Dictionary counting the variable expressions
        col_name: Column name for plot description

    Returns:
    Tuple for the plot variables (fig, ax)"""
    logging.debug("Read-in plot labels and values")
    labels = distribution.keys()
    values = distribution.values()

    # Create a figure and ax object
    logging.debug("Create fig and ax objects")
    fig, ax = plt.subplots()

    # Create a Bar Plot
    logging.debug("Create a bar plot")
    ax.bar(labels, values)
    ax.set_xlabel(col_name)
    ax.set_ylabel("Count")
    ax.set_title(f"{col_name} distribution")

    # Save the Bar Plot to a variable
    logging.debug("Save fig,ax tuple to plot variable")
    plot_variable = (fig, ax)
    return plot_variable


def show_plot(plot_variable: tuple) -> None:
    """
    Visualize a given plot variable
    Args:
        plot_variable: Tuple for the plot variables (fig, ax)

    Returns:
        None
        Prints plot
    """
    logging.debug("Decode fig,ax tuple")
    fig, ax = plot_variable
    logging.debug("Show plot")
    plt.show()
