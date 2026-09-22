import tiktoken
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage

# Tokenizer for token estimation
TOKENIZER = tiktoken.get_encoding("cl100k_base")


def count_tokens_in_text(text: str) -> int:
    """Calculates token count for a given string."""
    if not text:
        return 0
    return len(TOKENIZER.encode(text))


def count_tokens_in_messages(messages: list[BaseMessage]) -> int:
    """Calculates total token count across a list of LangChain Message objects."""
    total_tokens = 0
    for msg in messages:
        total_tokens += 4  # Overhead per message
        if isinstance(msg.content, str):
            total_tokens += count_tokens_in_text(msg.content)
        elif isinstance(msg.content, list):
            for item in msg.content:
                if isinstance(item, dict) and "text" in item:
                    total_tokens += count_tokens_in_text(item["text"])
    return total_tokens


def summarize_chat_history(messages: list[BaseMessage], model) -> BaseMessage:
    """Calls the LLM to summarize a slice of older conversation messages."""
    formatted_messages = []
    for msg in messages:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        formatted_messages.append(f"{role}: {msg.content}")
    
    conversation_text = "\n".join(formatted_messages)

    summary_prompt = (
        "Summarize the following conversation history concisely into a brief paragraph. "
        "Retain all critical facts, technical requirements, and decisions made:\n\n"
        f"{conversation_text}"
    )

    response = model.invoke([HumanMessage(content=summary_prompt)])
    summary_content = response.content if isinstance(response.content, str) else str(response.content)
    return SystemMessage(content=f"Summary of previous context: {summary_content}")


def manage_context_budget(
    messages: list[BaseMessage], 
    model, 
    token_budget: int = 2000, 
    keep_recent_turns: int = 2
) -> tuple[list[BaseMessage], bool]:
    """Manages context budget and triggers auto-summarization if token budget is exceeded."""
    current_tokens = count_tokens_in_messages(messages)

    if current_tokens <= token_budget or len(messages) <= (keep_recent_turns + 1):
        return messages, False

    split_index = max(1, len(messages) - keep_recent_turns)
    old_messages = messages[:split_index]
    recent_messages = messages[split_index:]

    summary_msg = summarize_chat_history(old_messages, model)
    new_messages = [summary_msg] + recent_messages
    return new_messages, True