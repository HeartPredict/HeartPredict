import logging
from functools import lru_cache

import pandas as pd
from heartpredict.data import ProjectData
from heartpredict.enums import Column, CorrelationMethod
from typing_extensions import Self


class CorrelationBackend:
    def __init__(
            self, project_data: ProjectData
        ) -> None:
        self.df = project_data.df


    @classmethod
    @lru_cache
    def build(cls, project_data: ProjectData) -> Self:
        return cls(project_data)
    

    def get_column_correlation_to_death_event(
            self, column: Column, method: CorrelationMethod
        ) -> float:
        death_event = self.df[Column.DEATH_EVENT]

        try:
            correlation_column = self.df[column]
        except KeyError:
            logging.error("column not found")

        return correlation_column.corr(death_event, method=method) # type: ignore
    

    def get_correlation_matrix(self, method: CorrelationMethod) -> pd.DataFrame:
        return self.df.corr(method=method).round(2) # type: ignore
        


            
