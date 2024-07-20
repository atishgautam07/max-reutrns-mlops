from dataclasses import dataclass
# from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: str
    source_dir: str
    fetchRepo: bool
    calcTickers: bool
    bucketName: str


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: str
    data_path: str
    transfData: bool
    bucketName: str