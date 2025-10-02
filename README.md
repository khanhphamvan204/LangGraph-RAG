# 📚 API Documentation - FAISS Vector Management System

## 📋 Mục lục

- Vector Management Endpoints
- Query Processing Endpoints
- Models & Schemas
- Authentication
- Error Handling

---

## 🔐 Authentication

Tất cả endpoints yêu cầu Bearer token trong header:

```http
Authorization: Bearer <your_token>
```

Token được verify thông qua `verify_token` dependency.

---

## 📦 Vector Management Endpoints

### 1. Upload Document

Thêm tài liệu mới vào vector database.

**Endpoint:**
```http
POST /documents/vector/add
Content-Type: multipart/form-data
Authorization: Bearer <token>
```

**Request Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| file | File | ✅ | File upload (PDF, TXT, DOCX, CSV, XLSX, XLS) |
| uploaded_by | string | ✅ | Tên người upload |

**Response:**
```json
{
  "message": "Vector added successfully",
  "_id": "uuid-generated-id",
  "filename": "document.pdf",
  "file_path": "/path/to/file",
  "vector_db_path": "/path/to/vectordb",
  "status": "created"
}
```

**Error Codes:**
- 400: File format không hỗ trợ hoặc JSON không hợp lệ
- 409: File đã tồn tại
- 500: Lỗi xử lý embeddings

**Example:**
```python
import requests

files = {'file': open('document.pdf', 'rb')}
data = {'uploaded_by': 'Nguyễn Văn A'}
headers = {'Authorization': 'Bearer your_token'}

response = requests.post(
    'http://localhost:3636/documents/vector/add',
    files=files,
    data=data,
    headers=headers
)
```

---

### 2. Delete Document

Xóa tài liệu khỏi hệ thống (file, metadata, vector embeddings).

**Endpoint:**
```http
DELETE /documents/vector/{doc_id}
Authorization: Bearer <token>
```

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| doc_id | string | ✅ | ID của document cần xóa |

**Response:**
```json
{
  "message": "Document deleted successfully",
  "_id": "doc-id",
  "filename": "document.pdf",
  "deletion_results": {
    "file_deleted": true,
    "metadata_deleted": true,
    "vector_deleted": true
  }
}
```

**Partial Deletion Response:**
```json
{
  "message": "Document partially deleted",
  "_id": "doc-id",
  "filename": "document.pdf",
  "deletion_results": {
    "file_deleted": true,
    "metadata_deleted": false,
    "vector_deleted": true
  },
  "warning": "Some components could not be deleted"
}
```

---

### 3. Get Document Info

Lấy thông tin chi tiết của document.

**Endpoint:**
```http
GET /documents/vector/{doc_id}
Authorization: Bearer <token>
```

**Response:**
```json
{
  "_id": "doc-id",
  "filename": "document.pdf",
  "url": "/path/to/file",
  "uploaded_by": "Nguyễn Văn A",
  "createdAt": "2025-10-02T10:30:00+07:00",
  "file_exists": true,
  "vector_exists": true,
  "file_size": 1024000
}
```

---

### 4. Update Document

Cập nhật metadata và tùy chọn tái tạo embeddings.

**Endpoint:**
```http
PUT /documents/vector/{doc_id}
Content-Type: multipart/form-data
Authorization: Bearer <token>
```

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| doc_id | string | ✅ | ID của document cần update |

**Form Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| filename | string | ❌ | Tên file mới (không bao gồm extension) |
| uploaded_by | string | ❌ | Người upload mới |
| force_re_embed | boolean | ❌ | Bắt buộc tái tạo embeddings (default: false) |

**Important Notes:**
- ⚠️ Khi đổi tên file, không được thêm extension - extension gốc sẽ được giữ nguyên
- Nếu filename thay đổi → tự động re-embed
- Nếu chỉ đổi uploaded_by → chỉ update metadata (trừ khi force_re_embed=true)

**Response:**
```json
{
  "message": "Document updated successfully",
  "_id": "doc-id",
  "success": true,
  "updated_fields": {
    "filename": {
      "old": "old_document.pdf",
      "new": "new_document.pdf",
      "changed": true
    },
    "uploaded_by": {
      "old": "User A",
      "new": "User B",
      "changed": true
    }
  },
  "operations": {
    "file_renamed": true,
    "vector_updated": true,
    "metadata_updated": true,
    "update_method": "full_re_embed"
  },
  "paths": {
    "old_file_path": "/old/path",
    "new_file_path": "/new/path",
    "old_vector_db": "/old/vectordb",
    "new_vector_db": "/new/vectordb"
  },
  "updatedAt": "2025-10-02T11:00:00+07:00",
  "force_re_embed": false
}
```

**Error Codes:**
- 400: Filename có extension hoặc extension không hỗ trợ
- 404: Document không tồn tại
- 409: Tên file mới đã tồn tại
- 500: Lỗi update

---

## 🔍 Query Processing Endpoints

### 5. Vector Search

Tìm kiếm semantic trong vector database.

**Endpoint:**
```http
POST /documents/vector/search
Content-Type: application/json
Authorization: Bearer <token>
```

**Request Body:**
```json
{
  "query": "tìm kiếm nội dung liên quan",
  "k": 5,
  "similarity_threshold": 0.7
}
```

**Request Schema:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| query | string | ✅ | - | Câu truy vấn |
| k | integer | ❌ | 5 | Số kết quả trả về |
| similarity_threshold | float | ❌ | 0.0 | Ngưỡng similarity (0-1) |

**Response:**
```json
{
  "query": "tìm kiếm nội dung liên quan",
  "results": [
    {
      "content": "Nội dung document...",
      "metadata": {
        "_id": "doc-id",
        "filename": "document.pdf",
        "uploaded_by": "User A",
        "similarity_score": 0.85
      }
    }
  ],
  "total_found": 3,
  "k_requested": 5,
  "similarity_threshold": 0.7,
  "search_time_ms": 120.5
}
```

**Similarity Score:**
- Chuyển đổi từ L2 distance: `score = 1 / (1 + distance)`
- Range: [0, 1] (1 = giống nhất)
- Chỉ trả về results có score >= similarity_threshold

---

### 6. Search with LLM

Tìm kiếm và tạo câu trả lời tự nhiên bằng LLM.

**Endpoint:**
```http
POST /documents/vector/search-with-llm
Content-Type: application/json
Authorization: Bearer <token>
```

**Request Body:** (giống /search)
```json
{
  "query": "Giải thích về chủ đề X",
  "k": 3,
  "similarity_threshold": 0.75
}
```

**Response:**
```json
{
  "llm_response": "Dựa trên tài liệu:\n\n1. **Điểm chính 1**: ...\n2. **Điểm chính 2**: ...\n\n**Kết luận**: ..."
}
```

**LLM Prompt Template:**
- Chỉ dùng thông tin từ documents
- Format markdown với số thứ tự/gạch đầu dòng
- Không thêm kiến thức bên ngoài
- Trả lời "Không tìm thấy thông tin" nếu không có dữ liệu

---

### 7. Process Query (Agent)

Xử lý truy vấn thông minh với LangGraph agent (chọn tool phù hợp).

**Endpoint:**
```http
POST /documents/vector/process-query
Content-Type: application/json
Authorization: Bearer <token>
```

**Request Body:**
```json
{
  "query": "Tìm pizza hải sản size lớn",
  "thread_id": "optional-conversation-id"
}
```

**Request Schema:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| query | string | ✅ | Câu hỏi người dùng |
| thread_id | string | ❌ | ID cuộc hội thoại (để lưu context) |

**Response Types:**

**Type 1: Database Search (Product)**
```json
{
  "status": "success",
  "data": {
    "natural_response": "Tìm thấy 2 pizza hải sản size lớn:",
    "product_variants": [
      {
        "id": "variant-123",
        "product_id": 1,
        "product_name": "Pizza Hải Sản",
        "price": 150000,
        "size_name": "Large",
        "crust_name": "Mỏng giòn",
        "stock": 10,
        "product_image_url": "https://...",
        "category_name": "Seafood"
      }
    ],
    "search_type": "database"
  },
  "error": null,
  "thread_id": "uuid-thread-id"
}
```

**Type 2: RAG Search (Document)**
```json
{
  "status": "success",
  "data": {
    "answer": "Dựa trên tài liệu, câu trả lời là...",
    "search_type": "rag"
  },
  "error": null,
  "thread_id": "uuid-thread-id"
}
```

**Type 3: Direct Response (Greeting)**
```json
{
  "status": "success",
  "data": {
    "message": "Chào bạn! Tôi có thể giúp gì?",
    "search_type": "direct"
  },
  "error": null,
  "thread_id": "uuid-thread-id"
}
```

**Agent Flow:**
1. Phân tích query → chọn tool:
   - `product_search`: Tìm pizza/sản phẩm
   - `vector_rag_search`: Tìm thông tin tài liệu
   - Direct response: Chào hỏi, câu đơn giản

2. Gọi tool → lấy kết quả
3. Format response với Pydantic models
4. Lưu conversation history (nếu có thread_id)

---

## 📊 Models & Schemas

### VectorSearchRequest
```python
class VectorSearchRequest(BaseModel):
    query: str
    k: int = 5
    similarity_threshold: float = 0.0
```

### SearchResult
```python
class SearchResult(BaseModel):
    content: str
    metadata: dict
```

### VectorSearchResponse
```python
class VectorSearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_found: int
    k_requested: int
    similarity_threshold: float
    search_time_ms: float
```

### ProcessQueryResponse
```python
class ProcessQueryResponse(BaseModel):
    status: str  # "success" | "error"
    data: Union[DatabaseResponse, RAGResponse, DirectResponse, None]
    error: Optional[str]
    thread_id: Optional[str]
```

### DatabaseResponse
```python
class DatabaseResponse(BaseModel):
    natural_response: str
    product_variants: List[dict]  # Flexible dict structure
    search_type: str = "database"
```

### RAGResponse
```python
class RAGResponse(BaseModel):
    answer: str
    search_type: str = "rag"
```

---

## ⚠️ Error Handling

### Common Error Responses
```json
{
  "detail": "Error message description"
}
```

### HTTP Status Codes

| Code | Meaning | Example |
|------|---------|---------|
| 200 | Success | Request processed successfully |
| 400 | Bad Request | Invalid file format, JSON parse error |
| 401 | Unauthorized | Missing/invalid token |
| 404 | Not Found | Document ID không tồn tại |
| 409 | Conflict | File đã tồn tại |
| 500 | Internal Server Error | Database/embedding error |

---

## 🔧 Configuration

### Environment Variables
```env
GEMINI_API_KEY=your_gemini_api_key
GOOGLE_API_KEY=your_google_api_key  # For LangChain
DATABASE_URL=mongodb://localhost:27017/
DATA_PATH=Root_Folder
VECTOR_DB_PATH=vectorstore
```

### Supported File Types
- 📄 PDF (.pdf)
- 📝 Text (.txt)
- 📘 Word (.docx)
- 📊 CSV (.csv)
- 📈 Excel (.xlsx, .xls)

---

## 💡 Best Practices

### Upload Documents:
- Đặt tên file rõ ràng, không dấu tiếng Việt
- File < 50MB để tránh timeout
- Sử dụng PDF OCR-enabled cho scan documents

### Vector Search:
- Đặt similarity_threshold cao (0.7-0.8) cho kết quả chính xác
- Tăng k nếu cần nhiều context cho LLM
- Query ngắn gọn, rõ ràng (5-15 từ)

### Update Documents:
- Backup trước khi update
- Sử dụng force_re_embed=true sau khi sửa nội dung file
- Không thêm extension vào filename parameter

### Agent Query:
- Dùng thread_id để duy trì context cuộc hội thoại
- Câu hỏi cụ thể cho kết quả tốt hơn
- Kiểm tra search_type trong response để xử lý phù hợp

---

## 📞 Support

- **Documentation:** Full README
- **Issues:** GitHub Issues
---

## 🐳 Docker Deployment

### Docker Compose Configuration

Hệ thống hỗ trợ 2 mode triển khai:

#### Mode 1: Standalone (Không MongoDB)

Sử dụng khi không cần database hoặc đã có MongoDB server riêng.

```yaml
services:
  # FastAPI Application
  app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: faiss-api-main
    ports:
      - "3636:3636"
    volumes:
      - ./Root_Folder:/app/Root_Folder
      - ./.env:/app/.env
    environment:
      - PYTHONUNBUFFERED=1
    networks:
      - app-network
    restart: unless-stopped

networks:
  app-network:
    driver: bridge
```

#### Mode 2: Full Stack (FastAPI + MongoDB)

Triển khai cả API và MongoDB trong cùng một stack.

```yaml
services:
  # FastAPI Application
  app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: faiss-api-main
    ports:
      - "3636:3636"
    volumes:
      - ./Root_Folder:/app/Root_Folder
      - ./.env:/app/.env
    environment:
      - PYTHONUNBUFFERED=1
    networks:
      - app-network
    depends_on:
      mongo:
        condition: service_healthy
    restart: unless-stopped

  # MongoDB Database
  mongo:
    image: mongo:6.0
    container_name: mongo-db
    ports:
      - "27017:27017"
    volumes:
      - mongo-data:/data/db
    environment:
      - MONGO_INITDB_ROOT_USERNAME=admin
      - MONGO_INITDB_ROOT_PASSWORD=123
      - MONGO_INITDB_DATABASE=faiss_db
    healthcheck:
      test: |
        mongosh --host localhost \
                --port 27017 \
                --username admin \
                --password 123 \
                --authenticationDatabase admin \
                --eval "db.adminCommand('ping')"
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 20s
    networks:
      - app-network
    restart: unless-stopped

volumes:
  mongo-data:
    driver: local

networks:
  app-network:
    driver: bridge
```

### Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Cài các gói hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-vie \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements và cài python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 3636

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3636"]
```

### Environment Variables (.env)

```env
# API Keys
GEMINI_API_KEY=your_gemini_api_key
GOOGLE_API_KEY=your_google_api_key

# MongoDB Configuration (for Full Stack mode)
DATABASE_URL=mongodb://admin:123@mongo:27017/

# Paths
DATA_PATH=Root_Folder
VECTOR_DB_PATH=vectorstore

# JWT Secret (optional)
JWT_SECRET_KEY=your-secret-key-here
```

### Deployment Commands

#### Khởi động hệ thống:

```bash
# Mode 1: Standalone
docker-compose up -d

# Mode 2: Full Stack (uncomment MongoDB service trước)
docker-compose up -d
```

#### Xem logs:

```bash
# Logs của API
docker-compose logs -f app

# Logs của MongoDB
docker-compose logs -f mongo

# Logs tất cả services
docker-compose logs -f
```

#### Dừng hệ thống:

```bash
docker-compose down

# Xóa luôn volumes (⚠️ Mất dữ liệu MongoDB)
docker-compose down -v
```

#### Rebuild image:

```bash
# Rebuild khi có thay đổi code
docker-compose up -d --build

# Rebuild không cache
docker-compose build --no-cache
docker-compose up -d
```

#### Vào container để debug:

```bash
# Vào FastAPI container
docker exec -it faiss-api-main bash

# Vào MongoDB container
docker exec -it mongo-db mongosh -u admin -p 123
```

### Volume Management

**Persistent Data:**
- `./Root_Folder`: Chứa uploaded files và vector databases
- `mongo-data`: Chứa MongoDB data (chỉ ở Full Stack mode)
- `.env`: Configuration file

**Important Notes:**
- ⚠️ Không xóa `Root_Folder` khi đang chạy
- 💾 Backup `Root_Folder` và `mongo-data` thường xuyên
- 🔒 `.env` file không nên commit lên Git

### Health Checks

#### API Health:
```bash
curl http://localhost:3636/docs
```

#### MongoDB Health (Full Stack mode):
```bash
docker exec mongo-db mongosh \
  -u admin -p 123 \
  --eval "db.adminCommand('ping')"
```

### Performance Tuning

#### Docker Resource Limits:

```yaml
services:
  app:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

#### MongoDB Optimization:

```yaml
mongo:
  command: mongod --wiredTigerCacheSizeGB 1.5
  deploy:
    resources:
      limits:
        memory: 2G
```

### Troubleshooting

**Lỗi thường gặp:**

1. **Port đã được sử dụng:**
   ```bash
   # Thay đổi port trong docker-compose.yml
   ports:
     - "8001:3636"  # Host:Container
   ```

2. **MongoDB không start:**
   ```bash
   # Check logs
   docker-compose logs mongo
   
   # Reset MongoDB data
   docker-compose down -v
   docker-compose up -d
   ```

3. **File không tìm thấy:**
   ```bash
   # Kiểm tra volume mount
   docker exec faiss-api-main ls -la /app/Root_Folder
   ```

4. **Permission denied:**
   ```bash
   # Fix ownership
   sudo chown -R $USER:$USER Root_Folder
   ```

### Production Deployment

**Checklist trước khi deploy:**

- [ ] Đổi MongoDB credentials mặc định
- [ ] Set `JWT_SECRET_KEY` phức tạp
- [ ] Enable HTTPS/SSL
- [ ] Setup backup automation
- [ ] Configure monitoring (Prometheus/Grafana)
- [ ] Setup log rotation
- [ ] Limit API rate limiting
- [ ] Use Docker secrets thay vì .env file