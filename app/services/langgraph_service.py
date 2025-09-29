from typing import Annotated, Optional, List, Any, Union
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool, StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
import os
import logging
import uuid
import json
from dotenv import load_dotenv

load_dotenv()

from app.services.tool_service import search_tool as product_search_tool  # Tool tìm kiếm sản phẩm
from app.services.tool_service import rag_search_tool  # Tool RAG vector search

# Cấu hình logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Biến môi trường
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# ================== MODELS ==================
class DatabaseResponse(BaseModel):
    natural_response: str
    product_variants: list[dict[str, Any]] = Field(default=[], description="Danh sách variants dưới dạng dict linh hoạt")
    search_type: str = "database"


class RAGResponse(BaseModel):
    answer: str
    search_type: str = "rag"


class DirectResponse(BaseModel):
    message: str
    search_type: str = "direct"


class Response(BaseModel):
    status: str
    data: Union[DatabaseResponse, RAGResponse, DirectResponse, None] = None
    error: Optional[str] = None
    thread_id: Optional[str] = None


# ================== STATE ==================
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


memory = MemorySaver()
model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
model = model.bind_tools([product_search_tool, rag_search_tool])


def invoke_model(model, messages):
    """Gọi model với custom system prompt nếu cần."""
    valid_messages = []
    for msg in messages:
        if isinstance(msg, AIMessage) and not msg.content and not getattr(msg, "tool_calls", []):
            logger.warning(f"Bỏ qua tin nhắn rỗng: {msg}")
            continue
        valid_messages.append(msg)
    
    # Thêm system prompt nếu chưa có (kiểm tra đầu messages)
    if not valid_messages or not isinstance(valid_messages[0], SystemMessage):
        system_prompt = """Bạn là trợ lý thông minh cho cửa hàng pizza. 
        - Nếu người dùng hỏi về sản phẩm pizza (giá, loại, tìm kiếm), hãy dùng tool 'product_search' để query database.
        - Nếu hỏi về tài liệu hoặc thông tin chung, dùng tool 'vector_rag_search'.
        - Trả lời tự nhiên bằng tiếng Việt, thân thiện.
        - Nếu không biết, nói 'Tôi sẽ kiểm tra thêm nhé!' và gọi tool phù hợp.
        - Luôn ưu tiên gọi tool nếu cần dữ liệu thực tế."""
        valid_messages.insert(0, SystemMessage(content=system_prompt))
        logger.info("Đã thêm system prompt mặc định.")
    
    logger.debug(f"Valid messages sent to model: {[msg.model_dump() for msg in valid_messages]}")
    return model.invoke(valid_messages)


# ================== AGENT NODE ==================
def agent_node(state: AgentState) -> AgentState:
    messages = state["messages"]
    if not messages or not isinstance(messages[-1], HumanMessage) or not messages[-1].content.strip():
        return {"messages": [AIMessage(content="Vui lòng nhập câu hỏi hợp lệ.")]}

    last_message = messages[-1].content.lower().strip()
    if last_message in ["chào", "hi", "hello", "chào bạn"]:
        direct_response = DirectResponse(
            message="Chào bạn! Hãy hỏi về pizza hoặc thông tin về cửa hàng, tôi sẽ giúp ngay!",
            search_type="direct"
        )
        return {"messages": [AIMessage(content=json.dumps(direct_response.model_dump(), ensure_ascii=False))]}

    cleaned_messages = [msg for msg in messages if not isinstance(msg, ToolMessage)]
    logger.debug(f"Cleaned messages sent to model: {[msg.model_dump() for msg in cleaned_messages]}")
    try:
        response = invoke_model(model, cleaned_messages)
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"Lỗi khi gọi model: {str(e)}", exc_info=True)
        return {"messages": [AIMessage(content=f"Lỗi khi gọi model: {str(e)}")]}


# ================== TOOL NODE ==================
def tool_node(state: AgentState) -> AgentState:
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", []) or []
    tool_messages = []

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        if tool_name == "product_search":
            tool_func = product_search_tool
        elif tool_name == "vector_rag_search":
            if "query" not in tool_args:
                for msg in reversed(state["messages"]):
                    if isinstance(msg, HumanMessage):
                        tool_args["query"] = msg.content
                        break
            tool_func = rag_search_tool
        else:
            error_response = DirectResponse(
                message=f"Tool không hỗ trợ: {tool_name}",
                search_type="direct"
            )
            tool_messages.append(
                ToolMessage(
                    content=json.dumps(error_response.model_dump(), ensure_ascii=False),
                    tool_call_id=tool_call["id"]
                )
            )
            continue

        try:
            result = tool_func.invoke(tool_args)
            if hasattr(result, "model_dump"):
                result = result.model_dump()
            logger.debug(f"Raw tool result: {result}")

            if tool_name == "product_search":
                natural_response = result.get("natural_response", str(result))
                product_variants = result.get("product_variants", [])
                if not isinstance(product_variants, list):
                    product_variants = []

                formatted_variants = []
                for variant in product_variants:
                    if isinstance(variant, dict):
                        # Map sang cấu trúc gần với schema gốc để tương thích
                        mapped_variant = {
    # Ưu tiên variant_id, fallback sang id
    "id": variant.get("variant_id") or variant.get("id") or None,

    # ID dạng số -> None nếu không có
    "product_id": variant.get("product_id", None),
    "size_id": variant.get("size_id", None),
    "crust_id": variant.get("crust_id", None),
    "category_id": variant.get("category_id", None),

    # Numeric field -> None nếu không có
    "price": variant.get("price", None),
    "stock": variant.get("stock", None),
    "size_diameter": variant.get("size_diameter", None),

    # String field -> mặc định chuỗi rỗng
    "product_name": variant.get("product_name", ""),
    "product_description": variant.get("description") 
                          or variant.get("product_description", ""),
    "product_image_url": variant.get("image_url") 
                        or variant.get("product_image_url", ""),
    "category_name": variant.get("category") 
                    or variant.get("category_name", ""),
    "size_name": variant.get("size") 
                or variant.get("size_name", ""),
    "crust_name": variant.get("crust") 
                 or variant.get("crust_name", ""),
    "crust_description": variant.get("crust_description", ""),
}

                        formatted_variants.append(mapped_variant)
                    else:
                        formatted_variants.append({"raw_data": str(variant)})

                # Fallback nếu validation fail (dù giờ linh hoạt hơn)
                try:
                    db_response = DatabaseResponse(
                        natural_response=natural_response,
                        product_variants=formatted_variants,
                        search_type="database"
                    )
                    content = json.dumps(db_response.model_dump(), ensure_ascii=False)
                except Exception as e:
                    logger.warning(f"Validation fail cho DatabaseResponse, dùng raw dict: {e}")
                    raw_data = {
                        "natural_response": natural_response,
                        "product_variants": formatted_variants,
                        "search_type": "database"
                    }
                    content = json.dumps(raw_data, ensure_ascii=False)

            elif tool_name == "vector_rag_search":
                if isinstance(result, dict) and "llm_response" in result:
                    answer = result["llm_response"]
                elif isinstance(result, dict) and "natural_response" in result:
                    answer = result["natural_response"]
                else:
                    answer = str(result)

                rag_response = RAGResponse(answer=answer, search_type="rag")
                content = json.dumps(rag_response.model_dump(), ensure_ascii=False)

            logger.info(f"Tool {tool_name} formatted response: {content[:200]}...")
            tool_messages.append(ToolMessage(content=content, tool_call_id=tool_call["id"]))

        except Exception as e:
            logger.error(f"Lỗi tool {tool_name}: {str(e)}")
            error_response = DirectResponse(
                message=f"Lỗi tool {tool_name}: {str(e)}",
                search_type="direct"
            )
            tool_messages.append(
                ToolMessage(
                    content=json.dumps(error_response.model_dump(), ensure_ascii=False),
                    tool_call_id=tool_call["id"]
                )
            )

    return {"messages": tool_messages}


# ================== WORKFLOW ==================
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END


workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", END)

graph = workflow.compile(checkpointer=memory)


# ================== MAIN HANDLER ==================
def process_query(query: str, thread_id: str = None) -> str:
    if not query.strip():
        response = Response(
            status="error",
            data=None,
            error="Vui lòng nhập câu hỏi hợp lệ.",
            thread_id=thread_id
        )
        return json.dumps(response.model_dump(), ensure_ascii=False)

    if thread_id is None:
        thread_id = str(uuid.uuid4())

    config = {"configurable": {"thread_id": thread_id}}
    input_msg = HumanMessage(content=query)

    try:
        current_state = graph.get_state(config)
        if current_state.values and "messages" in current_state.values:
            messages = current_state.values["messages"]
        else:
            messages = []
    except Exception as e:
        logger.warning(f"Không thể lấy state hiện tại: {e}. Bắt đầu mới.")
        messages = []

    state = {"messages": messages + [input_msg]}

    try:
        final_messages = []
        for step in graph.stream(state, config, stream_mode="values"):
            if "messages" in step:
                final_messages = step["messages"]
                logger.debug(f"Step messages: {[type(msg).__name__ for msg in final_messages[-3:]]}")

        response_data = None

        if final_messages:
            for msg in reversed(final_messages):
                if isinstance(msg, ToolMessage) and msg.content:
                    try:
                        tool_data = json.loads(msg.content)
                        search_type = tool_data.get("search_type", "direct")
                        if search_type == "database":
                            # Fallback nếu validation fail
                            try:
                                response_data = DatabaseResponse(
                                    natural_response=tool_data.get("natural_response", ""),
                                    product_variants=tool_data.get("product_variants", []),
                                    search_type="database"
                                )
                            except Exception as e:
                                logger.warning(f"Validation fail trong process_query, dùng raw: {e}")
                                response_data = {
                                    "natural_response": tool_data.get("natural_response", ""),
                                    "product_variants": tool_data.get("product_variants", []),
                                    "search_type": "database"
                                }
                        elif search_type == "rag":
                            response_data = RAGResponse(
                                answer=tool_data.get("answer", ""),
                                search_type="rag"
                            )
                        else:
                            response_data = DirectResponse(
                                message=tool_data.get("message", str(tool_data)),
                                search_type="direct"
                            )
                        break
                    except json.JSONDecodeError:
                        response_data = DirectResponse(
                            message=msg.content,
                            search_type="direct"
                        )
                        break

                elif isinstance(msg, AIMessage) and msg.content:
                    try:
                        ai_data = json.loads(msg.content)
                        search_type = ai_data.get("search_type", "direct")
                        if search_type == "direct":
                            response_data = DirectResponse(
                                message=ai_data.get("message", msg.content),
                                search_type="direct"
                            )
                        elif search_type == "rag":
                            response_data = RAGResponse(
                                answer=ai_data.get("answer", msg.content),
                                search_type="rag"
                            )
                        elif search_type == "database":
                            response_data = DatabaseResponse(
                                natural_response=ai_data.get("natural_response", msg.content),
                                product_variants=ai_data.get("product_variants", []),
                                search_type="database"
                            )
                        else:
                            response_data = DirectResponse(
                                message=msg.content,
                                search_type="direct"
                            )
                        break
                    except json.JSONDecodeError:
                        response_data = DirectResponse(
                            message=msg.content,
                            search_type="direct"
                        )
                        break

        if not response_data:
            response_data = DirectResponse(message="Không có kết quả.", search_type="direct")

        # Nếu data là dict (fallback), return trực tiếp JSON dict để tránh validation nested
        if isinstance(response_data, dict):
            full_response = {
                "status": "success",
                "data": response_data,
                "error": None,
                "thread_id": thread_id
            }
            return json.dumps(full_response, ensure_ascii=False)
        else:
            response = Response(status="success", data=response_data, error=None, thread_id=thread_id)
            return json.dumps(response.model_dump(), ensure_ascii=False)

    except Exception as e:
        logger.error(f"Lỗi khi xử lý query: {str(e)}", exc_info=True)
        response = Response(status="error", data=None, error=f"Lỗi khi xử lý: {str(e)}", thread_id=thread_id)
        return json.dumps(response.model_dump(), ensure_ascii=False)



# ================== CONVERSATION HISTORY ==================
def get_conversation_history(thread_id: str) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = graph.get_state(config)
        if state.values and "messages" in state.values:
            messages = [
                {
                    "type": type(msg).__name__,
                    "content": msg.content if hasattr(msg, "content") else str(msg),
                    "tool_calls": getattr(msg, "tool_calls", None)
                }
                for msg in state.values["messages"]
            ]
            response = Response(
                status="success",
                data=DirectResponse(message=json.dumps(messages, ensure_ascii=False), search_type="direct"),
                error=None,
                thread_id=thread_id
            )
        else:
            response = Response(
                status="success",
                data=DirectResponse(message="[]", search_type="direct"),
                error=None,
                thread_id=thread_id
            )
        return json.dumps(response.model_dump(), ensure_ascii=False)
    except Exception as e:
        logger.error(f"Lỗi khi lấy lịch sử: {e}")
        response = Response(
            status="error", data=None, error=f"Lỗi khi lấy lịch sử: {str(e)}", thread_id=thread_id
        )
        return json.dumps(response.model_dump(), ensure_ascii=False)


def clear_conversation_history(thread_id: str) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    try:
        graph.update_state(config, {"messages": []})
        response = Response(
            status="success",
            data=DirectResponse(message=f"Đã xóa lịch sử cho thread_id: {thread_id}", search_type="direct"),
            error=None,
            thread_id=thread_id
        )
        return json.dumps(response.model_dump(), ensure_ascii=False)
    except Exception as e:
        logger.error(f"Lỗi khi xóa lịch sử: {e}")
        response = Response(
            status="error", data=None, error=f"Lỗi khi xóa lịch sử: {str(e)}", thread_id=thread_id
        )
        return json.dumps(response.model_dump(), ensure_ascii=False)


def list_all_threads() -> str:
    logger.warning("MemorySaver không hỗ trợ liệt kê threads. Implement custom storage nếu cần.")
    response = Response(
        status="success",
        data=DirectResponse(message="MemorySaver không hỗ trợ liệt kê threads.", search_type="direct"),
        error=None,
        thread_id=None
    )
    return json.dumps(response.model_dump(), ensure_ascii=False)


# ================== DEMO ==================
if __name__ == "__main__":
    print("=== Chào mừng đến với Bot Pizza Shop ===")
    current_thread_id = None

    while True:
        print("\n=== Menu ===")
        print("1. Nhập câu hỏi (mới hoặc tiếp tục lịch sử)")
        print("2. Xem lịch sử cuộc trò chuyện")
        print("3. Xóa lịch sử cuộc trò chuyện")
        print("4. Thoát")

        choice = input("Chọn tùy chọn (1-4): ").strip()

        if choice == "1":
            query = input("Nhập câu hỏi (ví dụ: 'Tìm pizza hải sản'): ").strip()
            if not query:
                print(json.dumps(json.loads(process_query(query, current_thread_id)), indent=2, ensure_ascii=False))
                continue

            use_existing = input("Sử dụng lịch sử hiện tại? (y/n, mặc định n): ").strip().lower()
            if use_existing == "y" and current_thread_id:
                thread_id = current_thread_id
                print(f"Tiếp tục với thread_id: {thread_id}")
            else:
                thread_id = (
                    str(uuid.uuid4())
                    if not current_thread_id
                    else input(f"Nhập thread_id mới (nhấn Enter để tạo mới, mặc định {current_thread_id}): ").strip()
                    or str(uuid.uuid4())
                )
                current_thread_id = thread_id
                print(f"Tạo/Đặt thread_id mới: {current_thread_id}")

            result = process_query(query, thread_id=thread_id)
            print(f"Kết quả: {json.dumps(json.loads(result), indent=2, ensure_ascii=False)}")

        elif choice == "2":
            if not current_thread_id:
                print("Chưa có thread_id nào được tạo. Vui lòng nhập câu hỏi trước!")
            else:
                history = get_conversation_history(current_thread_id)
                print(f"Lịch sử cuộc trò chuyện (thread_id: {current_thread_id}):")
                print(json.dumps(json.loads(history), indent=2, ensure_ascii=False))

        elif choice == "3":
            if not current_thread_id:
                print("Chưa có thread_id nào được tạo. Vui lòng nhập câu hỏi trước!")
            else:
                confirm = input(f"Bạn có chắc muốn xóa lịch sử của thread_id {current_thread_id}? (y/n): ").strip().lower()
                if confirm == "y":
                    result = clear_conversation_history(current_thread_id)
                    print(json.dumps(json.loads(result), indent=2, ensure_ascii=False))
                    current_thread_id = None
                else:
                    print("Hủy xóa lịch sử.")

        elif choice == "4":
            print("Thoát chương trình. Tạm biệt!")
            break

        else:
            print("Lựa chọn không hợp lệ! Vui lòng chọn 1-4.")