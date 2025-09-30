from typing import Optional, List
from app.config import Config
from pydantic import BaseModel, Field

class AddVectorRequest(BaseModel):
    id: str = Field(alias="_id")
    filename: str
    url: str
    uploaded_by: str
    createdAt: str

class SearchResult(BaseModel):
    content: str
    metadata: dict

class VectorSearchRequest(BaseModel):
    query: str = Field(..., description="Câu truy vấn tìm kiếm")
    k: int = Field(default=5, ge=1, le=100, description="Số lượng kết quả trả về (1-100)")
    similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="Ngưỡng độ tương quan (0.0-1.0)")

class VectorSearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total_found: int
    k_requested: int
    similarity_threshold: float
    search_time_ms: float

class ProductVariantResponse(BaseModel):
    id: int
    product_id: int
    size_id: Optional[int]
    crust_id: Optional[int]
    price: float
    stock: int
    # Product info
    product_name: str
    product_description: Optional[str]
    product_image_url: Optional[str]
    category_id: Optional[int]
    category_name: Optional[str]
    # Size and crust info
    size_name: Optional[str]
    size_diameter: Optional[float]
    crust_name: Optional[str]
    crust_description: Optional[str]

class SearchResponse(BaseModel):
    product_variants: List[ProductVariantResponse]
    natural_response: str  # Câu trả lời tự nhiên bằng tiếng Việt
    sql_query: Optional[str] = None
    method_used: str  # "text_to_sql" or "fallback"
    error: Optional[str] = None
    search_type: str  # "database" để phân biệt với RAG

