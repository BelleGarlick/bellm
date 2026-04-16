from pydantic import BaseModel


class DownloadDatasetConfig(BaseModel):

    download_path: str

    processed_path: str
