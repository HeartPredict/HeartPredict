from typing import Callable

from heartpredict.backend.correlation import CorrelationBackend, CorrelationMethod
from heartpredict.data import ProjectData
from heartpredict.enums import Column


def test_column_correlation(project_data_func: Callable[..., ProjectData]) -> None:
    project_data = project_data_func()
    backend = CorrelationBackend.build(project_data)

    pearson_result = backend.get_column_correlation_to_death_event(
        Column.SERUM_CREATININE, CorrelationMethod.PEARSON
        )
    assert round(pearson_result, 10) == 0.3112813958

    kendall_result = backend.get_column_correlation_to_death_event(
        Column.SERUM_CREATININE, CorrelationMethod.KENDALL
        )
    assert round(kendall_result, 10) == 0.3317318406

    spearman_result = backend.get_column_correlation_to_death_event(
        Column.SERUM_CREATININE, CorrelationMethod.SPEARMAN
        )
    assert round(spearman_result, 10) == 0.3910769544


def test_correlation_matrix(project_data_func: Callable[..., ProjectData]) -> None:
    project_data = project_data_func()
    backend = CorrelationBackend.build(project_data)

    pearson_matrix = backend.get_correlation_matrix(CorrelationMethod.PEARSON)
    assert pearson_matrix.shape == (13, 13)
    assert pearson_matrix["DEATH_EVENT"].loc["DEATH_EVENT"] == 1.0
    assert pearson_matrix["DEATH_EVENT"].loc["serum_creatinine"] == 0.31 

    kendall_matrix = backend.get_correlation_matrix(CorrelationMethod.KENDALL)
    assert kendall_matrix.shape == (13, 13)
    assert kendall_matrix["DEATH_EVENT"].loc["DEATH_EVENT"] == 1.0 
    assert kendall_matrix["DEATH_EVENT"].loc["serum_creatinine"] == 0.33

    spearman_matrix = backend.get_correlation_matrix(CorrelationMethod.SPEARMAN)
    assert spearman_matrix.shape == (13, 13)
    assert spearman_matrix["DEATH_EVENT"].loc["DEATH_EVENT"] == 1.0 
    assert spearman_matrix["DEATH_EVENT"].loc["serum_creatinine"] == 0.39