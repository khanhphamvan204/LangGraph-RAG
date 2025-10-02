from fastapi.params import Depends
from pydantic import BaseModel
import os
import logging
import re
from typing import List, Dict, Any, Optional
import mysql.connector
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_core.tools import StructuredTool
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app.models.vector_models import (VectorSearchRequest, ProductVariantResponse, SearchResponse)
from app.services.embedding_service import get_embedding_model
from app.services.file_service import get_file_paths



load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def standardization(distance: float) -> float:
    """Chuyển đổi khoảng cách L2 thành điểm tương đồng (similarity score) trong khoảng [0, 1]."""
    if distance < 0:
        return 0.0
    else:
        return 1 / (1 + distance)

# Pydantic models


class TextToSqlService:
    def __init__(self):
        self.api_key = os.getenv('GOOGLE_API_KEY')
        if self.api_key:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=self.api_key,
                temperature=0.1,
            )
        else:
            self.llm = None
        self.db_schema = self.get_db_schema()
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 3306)),
            'user': os.getenv('DB_USER', 'root'),
            'password': os.getenv('DB_PASSWORD', '1234'),
            'database': os.getenv('DB_NAME', 'pizza_shop'),
            'charset': 'utf8mb4',
            'autocommit': True
        }

    def get_db_connection(self):
        """Create and return a new database connection."""
        try:
            connection = mysql.connector.connect(**self.db_config)
            logger.info("MySQL connection established")
            return connection
        except Exception as e:
            logger.error(f"Failed to connect to MySQL: {e}")
            raise Exception("Database connection failed")

    def search_products(self, natural_language_query: str) -> SearchResponse:
        """Tìm kiếm biến thể sản phẩm bằng natural language."""
        connection = None  # Khởi tạo để tránh lỗi trong except
        try:
            # Kết nối DB trước khi thực hiện bất kỳ logic nào
            connection = self.get_db_connection()
            try:
                if not self.llm:
                    logger.warning('Google API key not configured or LLM not initialized, using fallback search')
                    variants = self.fallback_search(natural_language_query, connection)
                    natural_response = self.generate_fallback_response(natural_language_query, variants)
                    return SearchResponse(
                        product_variants=variants,
                        natural_response=natural_response,
                        method_used="fallback",
                        search_type="database",
                        error="API key not configured or LLM not initialized"
                    )

                sql = self.generate_sql(natural_language_query)
                logger.info(f'Generated SQL: {sql or "NULL"}')

                if not sql:
                    variants = self.fallback_search(natural_language_query, connection)
                    natural_response = self.generate_fallback_response(natural_language_query, variants)
                    return SearchResponse(
                        product_variants=variants,
                        natural_response=natural_response,
                        method_used="fallback",
                        search_type="database",
                        error="Failed to generate SQL"
                    )

                # Validate SQL trước khi thực thi
                if not self.validate_sql_security(sql):
                    logger.warning(f'SQL failed security validation: {sql}')
                    variants = self.fallback_search(natural_language_query, connection)
                    natural_response = self.generate_fallback_response(natural_language_query, variants)
                    return SearchResponse(
                        product_variants=variants,
                        natural_response=natural_response,
                        method_used="fallback",
                        search_type="database",
                        error="Generated SQL failed security validation"
                    )

                # Thực thi SQL
                results = self.execute_sql(sql, connection)
                if not results:
                    variants = self.fallback_search(natural_language_query, connection)
                    natural_response = self.generate_natural_response(natural_language_query, variants)
                    return SearchResponse(
                        product_variants=variants,
                        natural_response=natural_response,
                        sql_query=sql,
                        method_used="fallback" if not variants else "text_to_sql",
                        search_type="database",
                        error="SQL execution returned no results" if not variants else None
                    )

                # Lấy variant IDs từ kết quả SQL
                variant_ids = list(set([row['id'] for row in results if 'id' in row]))
                
                # Lấy thông tin chi tiết biến thể sản phẩm
                variants = self.get_variants_with_details(variant_ids, connection)

                # Tạo câu trả lời tự nhiên
                natural_response = self.generate_natural_response(natural_language_query, variants)

                return SearchResponse(
                    product_variants=variants,
                    natural_response=natural_response,
                    sql_query=sql,
                    method_used="text_to_sql",
                    search_type="database"
                )

            finally:
                if connection:
                    connection.close()
                logger.info("MySQL connection closed")

        except Exception as e:
            logger.error(f'Text-to-SQL error: {str(e)}')
            variants = self.fallback_search(natural_language_query, connection) if connection else []
            natural_response = self.generate_fallback_response(natural_language_query, variants)
            return SearchResponse(
                product_variants=variants,
                natural_response=natural_response,
                method_used="fallback",
                search_type="database",
                error=f"Error occurred: {str(e)}"
            )

    def generate_natural_response(self, query: str, variants: List[ProductVariantResponse]) -> str:
        """Tạo câu trả lời tự nhiên bằng tiếng Việt"""
        try:
            if not self.llm:
                return self.generate_fallback_response(query, variants)
            
            if not variants:
                return f"Xin lỗi, cửa hàng hiện tại không có sản phẩm nào phù hợp với yêu cầu '{query}' của bạn."

            # Chuẩn bị thông tin sản phẩm để truyền vào prompt
            products_info = []
            for variant in variants:
                info = f"- {variant.product_name}"
                if variant.size_name:
                    info += f" (size {variant.size_name})"
                if variant.crust_name:
                    info += f" - đế {variant.crust_name}"
                info += f" - Giá: {int(variant.price):,}đ"
                if variant.stock > 0:
                    info += f" - Còn {variant.stock} sản phẩm"
                else:
                    info += " - Hết hàng"
                products_info.append(info)

            products_text = "\n".join(products_info[:15])  

            prompt_template = PromptTemplate(
                input_variables=["query", "products", "count"],
                template="""Bạn là nhân viên tư vấn của cửa hàng pizza thân thiện và nhiệt tình. 
Khách hàng vừa hỏi: "{query}"

Cửa hàng có {count} sản phẩm phù hợp:
{products}

Hãy trả lời khách hàng bằng tiếng Việt một cách tự nhiên, thân thiện và NGẮN GỌN. 
KHÔNG liệt kê chi tiết từng sản phẩm vì danh sách đã được trả về riêng.
Chỉ cần thông báo có bao nhiều sản phẩm phù hợp và khoảng giá tổng quát.
Nếu có sản phẩm hết hàng, chỉ cần thông báo ngắn gọn.
Kết thúc bằng câu hỏi tương tác ngắn.

Trả lời ngắn gọn:"""
            )

            chain = prompt_template | self.llm | StrOutputParser()
            response = chain.invoke({
                "query": query,
                "products": products_text,
                "count": len(variants)
            })

            return response.strip() if response else self.generate_fallback_response(query, variants)

        except Exception as e:
            logger.error(f'Error generating natural response: {str(e)}')
            return self.generate_fallback_response(query, variants)

    def generate_fallback_response(self, query: str, variants: List[ProductVariantResponse]) -> str:
        """Tạo câu trả lời fallback khi không có LLM"""
        if not variants:
            return f"Xin lỗi, cửa hàng hiện tại không có sản phẩm nào phù hợp với '{query}'. Bạn có thể thử từ khóa khác không?"

        count = len(variants)
        price_range = f"từ {int(min(v.price for v in variants)):,}đ đến {int(max(v.price for v in variants)):,}đ"
        
        # Kiểm tra có sản phẩm hết hàng không
        out_of_stock = any(v.stock == 0 for v in variants)
        in_stock_count = sum(1 for v in variants if v.stock > 0)
        
        if count == 1:
            variant = variants[0]
            stock_text = "còn hàng" if variant.stock > 0 else "đã hết hàng"
            return f"Có {count} sản phẩm phù hợp với giá {int(variant.price):,}đ, hiện tại {stock_text}. Bạn có muốn đặt không?"
        else:
            response = f"Có {count} sản phẩm phù hợp với giá {price_range}. "
            if out_of_stock and in_stock_count > 0:
                response += f"Trong đó {in_stock_count} sản phẩm còn hàng. "
            elif out_of_stock:
                response += "Tuy nhiên hiện tại đều đã hết hàng. "
            response += "Bạn có muốn xem chi tiết không?"
            return response

    def generate_sql(self, query: str) -> Optional[str]:
        """Tạo SQL từ natural language"""
        try:
            prompt_template = PromptTemplate(
                input_variables=["query", "schema"],
                template="""Convert Vietnamese query to SQL for MySQL database. Return ONLY the SQL query.

Schema: {schema}

Query format: SELECT pv.id FROM product_variants pv JOIN products p ON pv.product_id=p.id [JOINs] WHERE [conditions];
Important: 
- If having lots of results, LIMIT 15
- Use MySQL syntax and functions
- Use LIKE for text matching with % wildcards
- Always return pv.id (product_variants.id) in SELECT
- Focus on product_variants table as main table

Query: "{query}"

SQL:"""
            )

            chain = prompt_template | self.llm | StrOutputParser()
            sql_query = chain.invoke({
                "query": query,
                "schema": self.db_schema
            })

            if not sql_query:
                logger.warning('No SQL generated from LLM')
                return None

            # Làm sạch SQL query
            sql_query = self.clean_sql_query(sql_query)
            return sql_query

        except Exception as e:
            logger.error(f'Generate SQL error: {str(e)}')
            return None

    def get_db_schema(self) -> str:
        """Lấy database schema (rút gọn)"""
        return """
        product_variants(id,product_id,size_id,crust_id,price,stock) - Main table
        products(id,name,description,image_url,category_id) - Product info
        categories(id,name,description) - Product categories (pizza, salad, nước uống, etc.)
        sizes(id,name,diameter) - Pizza sizes
        crusts(id,name,description) - Pizza crust types
        """

    def clean_sql_query(self, sql: str) -> str:
        """Làm sạch SQL query"""
        # Loại bỏ markdown formatting
        sql = re.sub(r'```sql\n?', '', sql)
        sql = re.sub(r'```\n?', '', sql)
        sql = re.sub(r'`+', '', sql)

        # Loại bỏ comments
        sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
        sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)

        # Loại bỏ các dòng trống và trim
        sql = re.sub(r'\n\s*\n', '\n', sql)
        sql = sql.strip()

        # Đảm bảo kết thúc bằng semicolon
        if not sql.endswith(';'):
            sql += ';'

        return sql

    def validate_sql_security(self, sql: str) -> bool:
        """Validate SQL query để đảm bảo chỉ là SELECT"""
        if not sql:
            return False

        # Chuyển về lowercase để kiểm tra
        sql_lower = sql.lower().strip()

        # Chỉ cho phép SELECT queries
        if not re.match(r'^\s*select\s+', sql_lower):
            logger.warning(f'Non-SELECT query detected: {sql}')
            return False

        return True

    def execute_sql(self, sql: str, connection) -> List[Dict]:
        """Thực thi SQL query"""
        if not connection:
            logger.error("Database connection not available")
            return []

        try:
            cursor = connection.cursor(dictionary=True)
            try:
                # Validate SQL syntax
                cursor.execute(f"EXPLAIN {sql.rstrip(';')}")
                cursor.fetchall()
            except Exception as e:
                logger.warning(f'SQL syntax validation failed: {str(e)}')
                cursor.close()
                return []

            cursor.execute(sql)
            results = cursor.fetchall()
            cursor.close()
            
            logger.info(f'SQL executed successfully, returned {len(results)} rows')
            return results

        except Exception as e:
            logger.error(f'SQL execution error: {str(e)}')
            if 'cursor' in locals():
                cursor.close()
            return []

    def get_variants_with_details(self, variant_ids: List[int], connection) -> List[ProductVariantResponse]:
        """Lấy thông tin chi tiết biến thể sản phẩm với relations"""
        if not connection or not variant_ids:
            return []

        try:
            cursor = connection.cursor(dictionary=True)
            variant_ids_str = ','.join([str(id) for id in variant_ids])
            variants_query = f"""
                SELECT 
                    pv.id, pv.product_id, pv.size_id, pv.crust_id, 
                    pv.price, pv.stock,
                    p.name as product_name,
                    p.description as product_description,
                    p.image_url as product_image_url,
                    p.category_id,
                    c.name as category_name,
                    s.name as size_name,
                    s.diameter as size_diameter,
                    cr.name as crust_name,
                    cr.description as crust_description
                FROM product_variants pv
                JOIN products p ON pv.product_id = p.id
                LEFT JOIN categories c ON p.category_id = c.id
                LEFT JOIN sizes s ON pv.size_id = s.id
                LEFT JOIN crusts cr ON pv.crust_id = cr.id
                WHERE pv.id IN ({variant_ids_str})
                ORDER BY pv.price ASC
            """
            cursor.execute(variants_query)
            rows = cursor.fetchall()

            variants = []
            for row in rows:
                variant = ProductVariantResponse(
                    id=row['id'],
                    product_id=row['product_id'],
                    size_id=row['size_id'],
                    crust_id=row['crust_id'],
                    price=float(row['price']) if row['price'] else 0.0,
                    stock=row['stock'] if row['stock'] else 0,
                    product_name=row['product_name'],
                    product_description=row['product_description'],
                    product_image_url=row['product_image_url'],
                    category_id=row['category_id'],
                    category_name=row['category_name'],
                    size_name=row['size_name'],
                    size_diameter=float(row['size_diameter']) if row['size_diameter'] else None,
                    crust_name=row['crust_name'],
                    crust_description=row['crust_description']
                )
                variants.append(variant)

            cursor.close()
            return variants

        except Exception as e:
            logger.error(f'Error getting variants with details: {str(e)}')
            if 'cursor' in locals():
                cursor.close()
            return []

    def fallback_search(self, query: str, connection) -> List[ProductVariantResponse]:
        """Fallback search khi text-to-SQL thất bại"""
        logger.info(f'Using fallback search for query: {query}')

        if not connection:
            return []

        try:
            search_terms = [term.lower() for term in query.split() if len(term) > 2]
            
            if not search_terms:
                return []

            cursor = connection.cursor(dictionary=True)
            conditions = []
            params = []

            for term in search_terms:
                like_term = f"%{term}%"
                conditions.append("""
                    (p.name LIKE %s OR p.description LIKE %s OR c.name LIKE %s 
                     OR s.name LIKE %s OR cr.name LIKE %s)
                """)
                params.extend([like_term, like_term, like_term, like_term, like_term])

            where_clause = " OR ".join(conditions) if conditions else "1=1"

            fallback_query = f"""
                SELECT 
                    pv.id, pv.product_id, pv.size_id, pv.crust_id, 
                    pv.price, pv.stock,
                    p.name as product_name,
                    p.description as product_description,
                    p.image_url as product_image_url,
                    p.category_id,
                    c.name as category_name,
                    s.name as size_name,
                    s.diameter as size_diameter,
                    cr.name as crust_name,
                    cr.description as crust_description
                FROM product_variants pv
                JOIN products p ON pv.product_id = p.id
                LEFT JOIN categories c ON p.category_id = c.id
                LEFT JOIN sizes s ON pv.size_id = s.id
                LEFT JOIN crusts cr ON pv.crust_id = cr.id
                WHERE {where_clause}
                ORDER BY pv.price ASC
                LIMIT 20
            """

            cursor.execute(fallback_query, params)
            rows = cursor.fetchall()
            cursor.close()

            variants = []
            for row in rows:
                variant = ProductVariantResponse(
                    id=row['id'],
                    product_id=row['product_id'],
                    size_id=row['size_id'],
                    crust_id=row['crust_id'],
                    price=float(row['price']) if row['price'] else 0.0,
                    stock=row['stock'] if row['stock'] else 0,
                    product_name=row['product_name'],
                    product_description=row['product_description'],
                    product_image_url=row['product_image_url'],
                    category_id=row['category_id'],
                    category_name=row['category_name'],
                    size_name=row['size_name'],
                    size_diameter=float(row['size_diameter']) if row['size_diameter'] else None,
                    crust_name=row['crust_name'],
                    crust_description=row['crust_description']
                )
                variants.append(variant)

            return variants

        except Exception as e:
            logger.error(f'Fallback search error: {str(e)}')
            if 'cursor' in locals():
                cursor.close()
            return []

# Initialize service
service = TextToSqlService()

search_tool = StructuredTool.from_function(
    func=service.search_products,
    name="product_search",
    description="Tìm kiếm sản phẩm trong cửa hàng pizza dựa trên câu hỏi bằng tiếng Việt. Trả về danh sách các biến thể sản phẩm phù hợp (kèm thông tin chi tiết như giá, kích thước, loại đế) và một câu trả lời tự nhiên bằng tiếng Việt."
)

class RAGResponse(BaseModel):
    llm_response: str  # Câu trả lời từ LLM
    search_type: str = "rag"  # Để phân biệt với database search

class RAGSearchService:
    def __init__(self):
        self.api_key = os.getenv('GOOGLE_API_KEY')
        if self.api_key:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=self.api_key,
                temperature=0.1,
            )
        else:
            self.llm = None

    def search_with_llm(self, request: VectorSearchRequest) -> RAGResponse:
        try:
            _, vector_db_path = get_file_paths("dummy_filename")

            # Check if vector DB exists
            if not (os.path.exists(f"{vector_db_path}/index.faiss") and os.path.exists(f"{vector_db_path}/index.pkl")):
                logger.warning("Vector DB not found at {vector_db_path}")
                return RAGResponse(llm_response="Xin lỗi, tôi không tìm thấy thông tin này.", search_type="rag")

            # Load vector DB
            try:
                embedding_model = get_embedding_model()
                db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
                logger.info(f"Vector DB loaded with {db.index.ntotal} documents")
            except Exception as e:
                logger.error(f"Failed to load vector database: {str(e)}")
                return RAGResponse(llm_response=f"Không thể tải vector database: {str(e)}", search_type="rag")

            # Perform similarity search
            try:
                docs_with_scores = db.similarity_search_with_score(
                    request.query,
                    k=request.k
                )
                logger.info(f"Raw search: {len(docs_with_scores)} docs, scores: {[score for _, score in docs_with_scores]}")

                # Chuẩn hóa và lọc theo threshold
                filtered_docs = [
                    (doc, standardization(score)) for doc, score in docs_with_scores
                ]

                # Chuyển sang dict
                search_results = [
                    {
                        "content": doc.page_content,
                        "metadata": {**doc.metadata, "similarity_score": float(score)}
                    }
                    for doc, score in filtered_docs
                ]

                # Lấy top k
                top_results = search_results[:request.k]

                # Generate LLM response
                llm_response = "Xin lỗi, tôi không tìm thấy thông tin này."
                if top_results:
                    try:
                        # Tạo mới LLM giống hàm 1
                        llm = ChatGoogleGenerativeAI(
                            model="gemini-2.5-flash",
                            google_api_key=os.getenv('GOOGLE_API_KEY'),
                            temperature=0.3  # Tăng từ 0.1 để linh hoạt hơn, như hàm 1 (không set explicit)
                        )
                        if not llm:
                            llm_response = "LLM không được cấu hình."
                        else:
                            context = "\n\n".join(
                                [f"Document {i+1}:\n{result['content']}" for i, result in enumerate(top_results)]
                            )
                            logger.info(f"Context length: {len(context)} chars")

                            prompt_template = PromptTemplate(
                                input_variables=["query", "context"],
                                template="""
    Bạn là trợ lý hữu ích trả lời dựa trên context được cung cấp. 
    Nếu context có thông tin liên quan (dù ít), hãy trả lời ngắn gọn bằng tiếng Việt. 
    Chỉ dùng câu "Xin lỗi, tôi không tìm thấy thông tin này." nếu context hoàn toàn không liên quan.

    Cấu trúc:
    - Tóm tắt ngắn gọn.
    - Bullet points nếu cần.

    Query: {query}

    Context:
    {context}
    Answer:"""
                            )

                            # Gọi giống hàm 1: format prompt và invoke trực tiếp
                            prompt = prompt_template.format(query=request.query, context=context)
                            llm_response = llm.invoke(prompt).content

                    except Exception as e:
                        logger.error(f"LLM response generation failed: {str(e)}")
                        llm_response = "Không thể tạo câu trả lời từ LLM."

                return RAGResponse(llm_response=llm_response, search_type="rag")

            except Exception as e:
                logger.error(f"Search execution failed: {str(e)}")
                return RAGResponse(llm_response=f"Tìm kiếm thất bại: {str(e)}", search_type="rag")

        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return RAGResponse(llm_response="Lỗi hệ thống.", search_type="rag")
# Initialize service
rag_service = RAGSearchService()

# Define the RAG search tool for LangGraph
rag_search_tool = StructuredTool.from_function(
    func=lambda request: rag_service.search_with_llm(VectorSearchRequest(**request)) if isinstance(request, dict) else rag_service.search_with_llm(request),
    name="vector_rag_search",
    description="Thực hiện RAG search trên vector database (FAISS) để tìm tài liệu tương tự và generate câu trả lời từ LLM (Gemini). Input là query văn bản, k (top results), similarity_threshold, và current_user (cho quyền truy cập). Trả về llm_response với câu trả lời dựa trên context từ tài liệu, hoặc thông báo lỗi nếu không có dữ liệu phù hợp."
)