from temp import DataFrameAnalyzer, DiscreteStatistics, BooleanStatistics
import pandas as pd

def test_data_frame_analyzer():
    """Test the DataFrameAnalyzer object"""
    actual_object = DataFrameAnalyzer()
    actual_boolean = actual_object.calculate_boolean_statistics(boolean_column="smoking")

    expected_len = 5000
    expected_zero = 3441 / expected_len
    expected_one = 1559 / expected_len
    expected_boolean = BooleanStatistics(name= "smoking",
                                        zero= expected_zero,
                                        one= expected_one) 
    
    assert actual_boolean.zero == expected_boolean.zero
    assert actual_boolean.one == expected_boolean.one
    assert actual_boolean.name == expected_boolean.name
