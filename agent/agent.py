import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import TypedDict, Sequence, Annotated
from dotenv import load_dotenv

load_dotenv()

from langgraph.types import interrupt
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain.agents import create_agent
from langchain_core.tools import tool

from db import get_checkpointer
from tools import (
    get_user_profile,
    upsert_user_profile,
    get_meal_history,
    search_nutrition_kb,
    get_nutrition_info,
    validate_against_rda,
    detect_goal_drift,
    score_meal_plan,
    log_meal,
    get_running_macros,
    detect_deficiencies,
    analyze_nutrition_patterns,
    track_streaks,
)


class NutriState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def get_llm(temperature: float = 0.3) -> ChatBedrockConverse:
    return ChatBedrockConverse(
        model=os.getenv(
            "BEDROCK_MODEL_ID", "global.anthropic.claude-haiku-4-5-20251001-v1:0"
        ),
        region_name=os.getenv("BEDROCK_REGION", "us-east-1"),
        temperature=temperature,
        max_tokens=1024,
    )


MEMORY_AGENT_PROMPT = """You are the Memory Agent for NutriMind.
You manage user profiles and retrieve meal history.

Rules:
- Ensure proper Greeting when first time interacting with the user
- For new users: collect age, weight (kg), height (cm), goal, and dietary restrictions
- Confirm details with the user before calling upsert_user_profile
- When showing meal history, summarise patterns clearly (avg calories, common foods)
- user_id is always 'USER_#01'"""

NUTRITION_AGENT_PROMPT = """You are the Nutrition RAG Agent for NutriMind.
You answer nutrition and dietary science questions using evidence-based guidelines.

Rules:
- ALWAYS call search_nutrition_kb first for any nutrition science question —
  dietary guidelines, RDA context, macronutrient science, deficiency causes,
  disease-diet relationships
- Call get_nutrition_info when asked about macros or calories for a specific food
- Use validate_against_rda to contextualise whether a nutrient amount is adequate
- Combine both sources: RAG gives the guideline context, USDA gives the food data
- Cite your sources — mention DGA, WHO, or NIH when the answer comes from the KB
- Never give medical diagnoses
- If the user asks about supplements or medications, recommend consulting a doctor"""

PLANNING_AGENT_PROMPT = """You are the Planning Agent for NutriMind.
You generate adaptive meal plans grounded in the user's actual eating behaviour.

Rules:
- ALWAYS call detect_goal_drift before generating any meal plan
- Use drift flags to shape the plan (if protein is low, prioritise protein-dense meals)
- After writing a plan, call score_meal_plan to evaluate it
- Only return plans that pass (score >= 7) — if a plan fails, revise and score again
- Keep meals realistic and achievable
- user_id is always 'USER_#01'"""

INTAKE_AGENT_PROMPT = """You are the Intake Agent for NutriMind.
You log meals and track daily nutrition totals.

Rules:
- Log the meal with the data the user provides
- After every log_meal call, follow up with get_running_macros to show updated daily totals
- Call detect_deficiencies when the user asks about nutritional status or gaps
- user_id is always 'USER_#01'"""

INSIGHT_AGENT_PROMPT = """You are the Insight Agent for NutriMind.
You surface long-term nutrition patterns and generate proactive health alerts.

Rules:
- Always call analyze_nutrition_patterns first, then track_streaks
- If medical_flag is True (3+ days under 1200 kcal): flag clearly, recommend healthcare provider
- If iron_concern is True: suggest iron-rich foods (red meat, lentils, spinach, tofu)
- If calorie_concern is True: note the pattern and ask about recent changes
- Celebrate logging streaks positively
- Frame all insights as observations, never diagnoses
- user_id is always 'USER_#01'"""

SUPERVISOR_PROMPT = """You are the supervisor of NutriMind, an AI nutrition assistant.
Route the user's message to the correct specialist tool.

- call_memory_agent    -> user sends a greeting, wants to set up/update their profile, or view profile/meal history
- call_nutrition_agent -> user asks a nutrition question, wants food data, or macro/calorie info
- call_planning_agent  -> user wants a meal plan or to review goal progress
- call_intake_agent    -> user wants to log a meal, see today's macros, or check running totals
- call_insight_agent   -> user asks about health trends, streaks, or long-term patterns

Always Synthesize the specialist's full answer back to the user as your final response."""


memory_agent = create_agent(
    model=get_llm(temperature=0.1),
    tools=[get_user_profile, upsert_user_profile, get_meal_history],
    system_prompt=MEMORY_AGENT_PROMPT,
)

nutrition_rag_agent = create_agent(
    model=get_llm(temperature=0.3),
    tools=[search_nutrition_kb, get_nutrition_info, validate_against_rda],
    system_prompt=NUTRITION_AGENT_PROMPT,
)

planning_agent = create_agent(
    model=get_llm(temperature=0.7),
    tools=[detect_goal_drift, score_meal_plan],
    system_prompt=PLANNING_AGENT_PROMPT,
)

intake_agent = create_agent(
    model=get_llm(temperature=0),
    tools=[log_meal, get_running_macros, detect_deficiencies],
    system_prompt=INTAKE_AGENT_PROMPT,
)

insight_agent = create_agent(
    model=get_llm(temperature=0.3),
    tools=[analyze_nutrition_patterns, track_streaks],
    system_prompt=INSIGHT_AGENT_PROMPT,
)


@tool
def call_memory_agent(query: str) -> str:
    """Set up or update the user's profile, or retrieve meal history."""
    result = memory_agent.invoke({"messages": [HumanMessage(content=query)]})
    return result["messages"][-1].content


@tool
def call_nutrition_agent(query: str) -> str:
    """Answer nutrition questions, food data, or macro/calorie lookups."""
    result = nutrition_rag_agent.invoke({"messages": [HumanMessage(content=query)]})
    return result["messages"][-1].content


@tool
def call_planning_agent(query: str) -> str:
    """Generate a meal plan or review goal progress."""
    result = planning_agent.invoke({"messages": [HumanMessage(content=query)]})
    return result["messages"][-1].content


@tool
def call_intake_agent(query: str) -> str:
    """Log a meal or check today's running macro totals."""
    result = intake_agent.invoke({"messages": [HumanMessage(content=query)]})
    return result["messages"][-1].content


@tool
def call_insight_agent(query: str) -> str:
    """Analyze long-term nutrition patterns, streaks, or health trends."""
    result = insight_agent.invoke({"messages": [HumanMessage(content=query)]})
    last = result["messages"][-1].content
    if "medical_flag" in str(last).lower() or "1200" in str(last):
        interrupt("Medical concern flagged. Awaiting human review.")
    return last


supervisor = create_agent(
    model=get_llm(temperature=0),
    tools=[
        call_memory_agent,
        call_nutrition_agent,
        call_planning_agent,
        call_intake_agent,
        call_insight_agent,
    ],
    system_prompt=SUPERVISOR_PROMPT,
)


builder = StateGraph(NutriState)
builder.add_node("supervisor", supervisor)
builder.add_edge(START, "supervisor")
builder.add_edge("supervisor", END)
try:
    _checkpointer = get_checkpointer()
except Exception as e:
    _checkpointer = None
    print(f"NOTE: No PostgreSQL connection ({e}). Running without persistent memory.")

compiled = builder.compile(checkpointer=_checkpointer)
