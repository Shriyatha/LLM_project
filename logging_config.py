from pydantic import BaseModel
from typing import Optional

class LoggingConfig(BaseModel):
    log_file_name: str = "app.log"
    min_log_level: str = "INFO"
    log_rotation: str = "00:00"  # Rotate at midnight
    log_compression: str = "zip"
    log_server_address: Optional[str] = "tcp://127.0.0.1:5555"
    enable_network_logging: bool = False