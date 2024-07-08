from typing import Callable

import pytest
from heartpredict.data import MLData, FeatureData
from heartpredict.backend.ml import MLBackend, PretrainedModel

from sklearn.metrics import root_mean_squared_error


def test_load_pretrained_classifiers_seed_42(
        ml_data_func: Callable[..., MLData]
) -> None:
    model_dir = "results/trained_models/classifier"
    data = ml_data_func()
    pretrained_model = PretrainedModel()

    best_decision_tree_score = 0.859
    pretrained_model.load_model(
        f"{model_dir}/DecisionTreeClassifier_model_{data.random_seed}.joblib"
    )
    assert pretrained_model.model.score(
        data.valid.x, data.valid.y
    ) == best_decision_tree_score

    best_random_forest_score = 0.992
    pretrained_model.load_model(
        f"{model_dir}/RandomForestClassifier_model_{data.random_seed}.joblib"
    )
    assert pretrained_model.model.score(
        data.valid.x, data.valid.y
    ) == best_random_forest_score

    best_knn_score = 0.977
    pretrained_model.load_model(
        f"{model_dir}/KNeighborsClassifier_model_{data.random_seed}.joblib"
    )
    assert pretrained_model.model.score(data.valid.x, data.valid.y) == best_knn_score

    best_lda_score = 0.839
    pretrained_model.load_model(
        f"{model_dir}/LinearDiscriminantAnalysis_model_{data.random_seed}.joblib"
    )
    assert pretrained_model.model.score(data.valid.x, data.valid.y) == best_lda_score

    best_qda_score = 0.829
    pretrained_model.load_model(
        f"{model_dir}/QuadraticDiscriminantAnalysis_model_{data.random_seed}.joblib"
    )
    assert pretrained_model.model.score(data.valid.x, data.valid.y) == best_qda_score

    with pytest.raises(FileNotFoundError) as exc_info:
        pretrained_model.load_model("CoolModel.joblib")
    assert (
            str(exc_info.value) == "[Errno 2] No such file or directory: "
                                   "'CoolModel.joblib'"
    )


def test_train_model_for_classification_seed_42(
        ml_data_func: Callable[..., MLData],
) -> None:
    data = ml_data_func(random_seed=42)
    backend = MLBackend(data)
    pretrained_model = PretrainedModel()
    path_to_best_model = backend.classification_for_different_classifiers().model_file

    best_model_accuracy = 0.992
    pretrained_model.load_model(path_to_best_model)
    assert (
            pretrained_model.model.score(backend.data.valid.x, backend.data.valid.y)
            == best_model_accuracy
    )


def test_load_pretrained_regressors_seed_42(
        ml_data_func: Callable[..., MLData]
) -> None:
    model_dir = "results/trained_models/regressor"
    data = ml_data_func()
    pretrained_model = PretrainedModel()

    best_logistic_regression_score = 0.386
    pretrained_model.load_model(
        f"{model_dir}/LogisticRegression_model_{data.random_seed}.joblib"
    )
    error = round(root_mean_squared_error(
        data.valid.y,
        pretrained_model.model.predict(data.valid.x)),
        3)
    assert error == best_logistic_regression_score

    best_logistic_regression_cv_score = 0.386
    pretrained_model.load_model(
        f"{model_dir}/LogisticRegressionCV_model_{data.random_seed}.joblib"
    )

    error = round(root_mean_squared_error(
        data.valid.y,
        pretrained_model.model.predict(data.valid.x)),
        3)
    assert error == best_logistic_regression_cv_score


def test_train_model_for_regressionn_seed_42(
        ml_data_func: Callable[..., MLData],
) -> None:
    data = ml_data_func(random_seed=42)
    backend = MLBackend(data)
    pretrained_model = PretrainedModel()
    path_to_best_model = backend.regression_for_different_regressors().model_file

    best_model_rmse = 0.386
    pretrained_model.load_model(path_to_best_model)
    error = round(root_mean_squared_error(
        data.valid.y,
        pretrained_model.model.predict(data.valid.x)),
        3)
    assert error == best_model_rmse


def test_predict_death_event(
        feature_data_func: Callable[..., FeatureData],
) -> None:
    data = feature_data_func()
    pretrained_model = PretrainedModel()

    pretrained_model.load_model(
        "results/trained_models/classifier/RandomForestClassifier_model_42.joblib"
    )
    result = pretrained_model.model.predict(data.feature_matrix)
    assert result[0] == 0
    assert result[1] == 1
    assert result[2] == 0

    pretrained_model.load_model(
        "results/trained_models/regressor/LogisticRegression_model_42.joblib"
    )
    result = pretrained_model.model.predict(data.feature_matrix)
    assert result[0] == 0
    assert result[1] == 1
    assert result[2] == 0
