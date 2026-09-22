from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- Role System Prompts ---
DEVELOPER_SYSTEM_PROMPT = """You are an expert Senior Software Architect and Developer Assistant.
Your objective is to provide highly technical, precise, and actionable engineering answers.

Guidelines:
- Prioritize technical definitions, architectural logic, edge cases, and code design patterns.
- Include clear, clean code snippets where relevant.
- Be direct, concise, and technically rigorous. Avoid generic high-level fluff.
"""

MANAGER_SYSTEM_PROMPT = """You are a Strategic Engineering Director and Product Advisor.
Your objective is to provide executive summaries, strategic roadmaps, and business-level explanations.

Guidelines:
- Focus on business impact, cost vs. benefit, project timelines, risk management, and ROI.
- DO NOT provide raw code snippets or line-by-line syntax unless specifically requested.
- Present solutions using executive summaries, structured bullet points, and key architectural tradeoffs.
"""

DEFAULT_SYSTEM_PROMPT = """You are a helpful, context-aware AI assistant.
Provide clear, accurate, and concise answers based on the conversation context.
"""


def get_system_prompt(role: str) -> str:
    """Returns the system prompt string based on the selected user role."""
    role_clean = role.lower().strip() if role else "developer"
    
    if role_clean == "developer":
        return DEVELOPER_SYSTEM_PROMPT
    elif role_clean == "manager":
        return MANAGER_SYSTEM_PROMPT
    else:
        return DEFAULT_SYSTEM_PROMPT


def get_prompt_template(role: str) -> ChatPromptTemplate:
    """
    Constructs a LangChain ChatPromptTemplate containing:
    1. Dynamic System Prompt based on role
    2. Placeholder for existing conversation history
    3. The incoming user question
    """
    system_prompt = get_system_prompt(role)

    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])