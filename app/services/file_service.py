import os
import logging
from fastapi import HTTPException
from app.config import Config

logger = logging.getLogger(__name__)

def get_file_paths(filename: str) -> tuple[str, str]:
    if not filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required"
        )

    base_path = Config.DATA_PATH
    try:
        file_folder = Config.FILE_PATHS["file_folder"]
        vector_folder = Config.FILE_PATHS["vector_folder"]
    except KeyError as e:
        logger.error(f"Config error: Missing key {str(e)} in FILE_PATHS")
        raise HTTPException(status_code=500, detail=f"Configuration error: Missing {str(e)} in FILE_PATHS")

    file_path = os.path.join(base_path, file_folder, filename).replace("\\", "/")
    vector_db_path = os.path.join(base_path, vector_folder).replace("\\", "/")

    logger.info(f"get_file_paths called with filename: {filename}")
    logger.info(f"Returning: file_path={file_path}, vector_db_path={vector_db_path}")
    return file_path, vector_db_path