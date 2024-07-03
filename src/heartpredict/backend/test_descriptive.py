from descriptive import DataFrameAnalyzer, DiscreteStatistics, BooleanStatistics
import pandas as pd


def test_calculate_boolean_statistics():
    """
    Test the calculate_boolean_statistics function
    from the DataFrameAnalyzer object
    """
    actual_object = DataFrameAnalyzer()
    actual_boolean = actual_object.calculate_boolean_statistics(boolean_column="smoking")

    expected_len = 5000
    expected_zero = 3441 / expected_len
    expected_one = 1559 / expected_len
    expected_boolean = BooleanStatistics(name="smoking",
                                        zero=expected_zero,
                                        one=expected_one) 
    
    assert actual_boolean.zero == expected_boolean.zero
    assert actual_boolean.one == expected_boolean.one
    assert actual_boolean.name == expected_boolean.name


def test_calculate_discrete_statistics():
    """
    Test the calculate_discrete_statistics function
    from the DataFrameAnalyzer object
    """
    actual_object = DataFrameAnalyzer()
    actual_discrete = actual_object.calculate_discrete_statistics(discrete_column="age")

    expected_mean = 60.288736400000005
    expected_std = 11.697242810508323
    expected_discrete = DiscreteStatistics(name="age",
                                           minimum=40.0,
                                           maximum=95.0,
                                           median=65.0,
                                           mean=expected_mean,
                                           standard_dev=expected_std)
    
    assert expected_discrete.mean == actual_discrete.mean
    assert expected_discrete.standard_dev == actual_discrete.standard_dev
    assert expected_discrete.name == actual_discrete.name


def test_create_conditional_dataset():
    """
    Test the create_conditional_dataset function
    from the DataFrameAnalyzer object
    """
    data = {
        "col1": [1,2,3,4,5],
        "col2": [6,7,8,9,10]
    }
    original_df = pd.DataFrame(data)
    condition = original_df["col1"] <= 3
    expected_result = original_df[condition].copy()
    
    actual_object = DataFrameAnalyzer()
    actual_result = actual_object.create_conditional_dataset(df=original_df,
                                                             col="col1",
                                                             num=3,
                                                             rel="<=")

    assert expected_result["col1"].equals(actual_result["col1"])


def test_save_variable_distribution():
    """
    Test the save_variable_distribution function
    from the DataFrameAnalyzer object
    """
    expected_bool = {"Not smoking": 3441,
                     "Is smoking":1559
                     }
    
    actual_object = DataFrameAnalyzer()
    actual_bool = actual_object.save_variable_distribution(column="smoking")

    assert expected_bool["Is smoking"] == actual_bool["Is smoking"]
    assert expected_bool["Not smoking"] == actual_bool["Not smoking"]
