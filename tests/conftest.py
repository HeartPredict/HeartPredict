from pathlib import Path
from typing import Callable

import pytest
from heartpredict.backend.data import MLData, ProjectData, FeatureData


@pytest.fixture
def project_data_func() -> Callable[..., ProjectData]:
    def _project_data_factory(
            csv_path: Path = Path("data/heart_failure_clinical_records.csv"),
    ) -> ProjectData:
        return ProjectData.build(csv_path)

    return _project_data_factory


@pytest.fixture
def feature_data_func(
        project_data_func: Callable[..., ProjectData],
) -> Callable[..., FeatureData]:
    def _feature_data_factory() -> FeatureData:
        return FeatureData.build(
            project_data_func("data/example_data_points.csv"),
            Path("results/scalers/used_scaler.joblib"),
        )

    return _feature_data_factory


@pytest.fixture
def ml_data_func(
        project_data_func: Callable[..., ProjectData],
) -> Callable[..., MLData]:
    def _ml_data_factory(test_size: float = 0.2, random_seed: int = 42) -> MLData:
        return MLData.build(project_data_func(), test_size, random_seed)

    return _ml_data_factory
