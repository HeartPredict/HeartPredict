from enum import Enum


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class CorrelationMethod(str, Enum):
    PEARSON = "pearson"
    KENDALL = "kendall"
    SPEARMAN = "spearman"


class BoolColumn(str, Enum):
    ANAEMIA = "anaemia"
    DIABETES = "diabetes"
    HIGH_BLOOD_PRESSURE = "high_blood_pressure"
    SEX = "sex"
    SMOKING = "smoking"
    DEATH_EVENT = "DEATH_EVENT"


class DiscreteColumn(str, Enum):
    AGE = "age"
    CREATININE_PHOSPHOKINASE = "creatinine_phosphokinase"
    EJECTION_FRACTION = "ejection_fraction"
    PLATELETS = "platelets"
    SERUM_CREATININE = "serum_creatinine"
    SERUM_SODIUM = "serum_sodium"
    TIME = "time"


class Column(str, Enum):
    AGE = "age"
    ANAEMIA = "anaemia"
    CREATININE_PHOSPHOKINASE = "creatinine_phosphokinase"
    DIABETES = "diabetes"
    EJECTION_FRACTION = "ejection_fraction"
    HIGH_BLOOD_PRESSURE = "high_blood_pressure"
    PLATELETS = "platelets"
    SERUM_CREATININE = "serum_creatinine"
    SERUM_SODIUM = "serum_sodium"
    SEX = "sex"
    SMOKING = "smoking"
    TIME = "time"
    DEATH_EVENT = "DEATH_EVENT"