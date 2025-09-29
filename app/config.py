from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    DATA_PATH = os.getenv("DATA_PATH", "Root_Folder")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    DATABASE_URL = os.getenv("DATABASE_URL", "mongodb://localhost:27017/")
    
    FILE_PATHS = {
        'file_folder': os.getenv("FILE_FOLDER", "Rag_Info/File_Folder"),
        'vector_folder': os.getenv("VECTOR_FOLDER", "Rag_Info/Faiss_Folder")
    }