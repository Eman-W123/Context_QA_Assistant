import streamlit as st
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage

from config import get_model, TOKEN_BUDGET
from utils.prompts import get_prompt_template
from utils.token_manager import count_tokens_in_messages, manage_context_budget

# --- Page Configuration ---
st.set_page_config(
    page_title="Context-Aware Q&A Assistant",
    page_icon="🤖",
    layout="wide"
)

# --- Pydantic Schema for Structured Output ---
class AnswerSchema(BaseModel):
    main_answer: str = Field(description="The primary direct answer or code solution to the prompt.")
    sources_or_logic: list[str] = Field(description="Key reasoning steps, technical principles, or architectural tradeoffs considered.")


# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []  # Holds LangChain BaseMessage objects


# --- Sidebar UI ---
st.sidebar.title("🛠️ Assistant Settings")

# 1. Role Selector (Dynamic System Prompt)
user_role = st.sidebar.selectbox(
    "Select Persona / Role",
    options=["Developer", "Manager"],
    index=0,
    help="Developer: Technical, code-heavy, precise logic.\nManager: Strategic summaries, timelines, business impact."
)

st.sidebar.divider()

# 2. Model Provider & Switcher
model_provider = st.sidebar.radio(
    "Model Provider",
    options=["gemini", "ollama"],
    format_func=lambda x: "Google Gemini (Cloud API)" if x == "gemini" else "Ollama (Local Offline)",
    index=0
)

if model_provider == "gemini":
    selected_model_name = st.sidebar.selectbox(
        "Gemini Model",
        options=["gemini-2.5-flash", "gemini-2.5-pro"],
        index=0
    )
else:
    selected_model_name = st.sidebar.text_input(
        "Ollama Model Name",
        value="llama3",
        help="Make sure this model is pulled in your local Ollama server (e.g. `ollama run llama3`)."
    )

st.sidebar.divider()

# 3. Live Token Budget Counter
current_tokens = count_tokens_in_messages(st.session_state.messages)
st.sidebar.subheader("📊 Context Engineering Budget")
st.sidebar.metric(
    label="Current Context Tokens",
    value=f"{current_tokens} / {TOKEN_BUDGET} tokens",
    delta="Within Budget" if current_tokens <= TOKEN_BUDGET else "Over Budget (Summarization Pending)",
    delta_color="normal" if current_tokens <= TOKEN_BUDGET else "inverse"
)
st.sidebar.progress(min(current_tokens / TOKEN_BUDGET, 1.0))

# Clear Chat Button
if st.sidebar.button("🗑️ Clear Conversation"):
    st.session_state.messages = []
    st.rerun()


# --- Main Application Header ---
st.title("🤖 Context-Aware Q&A Assistant")
st.caption("Built on LCEL • Structured Output • Dynamic Role Prompts • Context Auto-Summarization")

# --- Render Existing Chat Messages ---
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)


# --- Handle User Input & LCEL Orchestration ---
if user_input := st.chat_input("Ask a question..."):
    # 1. Display user message immediately
    with st.chat_message("user"):
        st.write(user_input)

    # Append new user message to session state
    st.session_state.messages.append(HumanMessage(content=user_input))

    # 2. Context Engineering Layer: Manage Token Budget & Auto-Summarize if needed
    try:
        base_llm = get_model(provider=model_provider, model_name=selected_model_name)
    except Exception as e:
        st.error(f"Failed to initialize model: {e}")
        st.stop()

    managed_history, was_summarized = manage_context_budget(
        messages=st.session_state.messages[:-1],  # History excluding current user turn
        model=base_llm,
        token_budget=TOKEN_BUDGET
    )

    if was_summarized:
        st.toast("⚠️ Context limit exceeded! Older conversation turns were automatically summarized.", icon="✂️")

    # Update session state history with auto-summarized version
    st.session_state.messages = managed_history + [HumanMessage(content=user_input)]

    # 3. Build LCEL Pipeline
    prompt_template = get_prompt_template(user_role)
    
    # Bind structured output parser
    structured_llm = base_llm.with_structured_output(AnswerSchema)
    
    # Complete LCEL Chain
    chain = prompt_template | structured_llm

    # 4. Stream & Execute Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking & structuring output..."):
            try:
                # Invoke LCEL chain with dynamic context
                response: AnswerSchema = chain.invoke({
                    "chat_history": managed_history,
                    "input": user_input
                })

                # Format structured response
                output_md = f"{response.main_answer}\n\n"
                if response.sources_or_logic:
                    output_md += "#### Key Reasoning & Logic Steps:\n"
                    for item in response.sources_or_logic:
                        output_md += f"- {item}\n"

                st.markdown(output_md)

                # Save AI response to session state
                st.session_state.messages.append(AIMessage(content=output_md))

            except Exception as err:
                st.error(f"Error during chain execution: {err}")

    # Rerun to refresh live token metrics in sidebar
    st.rerun()