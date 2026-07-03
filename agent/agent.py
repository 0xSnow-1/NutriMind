import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv()
from botocore.config import Config  
from langgraph.types import Command, interrupt
from langsmith import traceable
from typing import Literal
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langchain.agents import create_agent

from db import get_checkpointer
from state import NutriState, DecisionRouting
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


@traceable
def get_llm(temperature: float = 0.3) -> ChatBedrockConverse:
    return ChatBedrockConverse(
        model=os.getenv(
            "BEDROCK_MODEL_ID", "global.anthropic.claude-haiku-4-5-20251001-v1:0"
        ),
        region_name=os.getenv("BEDROCK_REGION", "us-east-1"),
        temperature=temperature,
        max_tokens=1024,
        config=Config(                       # <-- ADD THIS BLOCK
            connect_timeout=5,
            read_timeout=20,
            retries={"max_attempts": 1},
        ),
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
Route the user's message to exactly one specialist. Call exactly one. Never do the work yourself.

- memory_agent          -> user wants to set up/update their profile, or view profile/meal history
- nutrition_rag_agent   -> user asks a nutrition question, wants food data, or macro/calorie info
- planning_agent        -> user wants a meal plan or to review goal progress
- intake_agent          -> user wants to log a meal, see today's macros, or check running totals
- insight_agent         -> user asks about health trends, streaks, or long-term patterns
- FINISH                -> question is fully answered, no more agents needed

CRITICAL: If the last message in the conversation is already a response from a specialist
(not a new question from the user), you MUST choose FINISH. Do NOT route again."""


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

structured_output_supervisor = get_llm(temperature=0).with_structured_output(
    DecisionRouting
)

@traceable
def supervisor_node(state: NutriState) -> Command[Literal["memory_agent",
        "nutrition_rag_agent",
        "planning_agent",
        "intake_agent",
        "insight_agent"]]:
    decision = structured_output_supervisor.invoke(
        [SystemMessage(content=SUPERVISOR_PROMPT)] + list(state["messages"])
    )
    goto = decision.next_agent
    if goto == "FINISH":
        return Command(goto=END, update={"next_agent": 'FINISH'})
    return Command(
        goto=decision.next_agent,
        update={
            "messages": [HumanMessage(content=f"[Supervisor -> {decision.next_agent}]: {decision.task_description}")],
            "next_agent": decision.next_agent,
        },
    )

@traceable
def memory_node(state: NutriState) -> Command[Literal["supervisor"]]:
    result = memory_agent.invoke({"messages": state["messages"]})
    return Command(
        update={"messages": result["messages"]},
        goto="supervisor",
    )
    
    

@traceable
def nutrition_node(state: NutriState) -> Command[Literal["supervisor"]]:
    result = nutrition_rag_agent.invoke({"messages": state["messages"]})
    return Command(
        update={"messages": result["messages"]},
            goto="supervisor"
    )

@traceable
def planning_node(state: NutriState) -> Command[Literal["supervisor"]]:
    result = planning_agent.invoke({"messages": state["messages"]})
    return Command(
        update={"messages": result["messages"]},
        goto="supervisor",
    )

@traceable
def intake_node(state: NutriState) -> Command[Literal["supervisor"]]:
    result = intake_agent.invoke({"messages": state["messages"]})
    return Command(
        update={"messages": result["messages"]},
        goto="supervisor",
    )

@traceable
def insight_node(state: NutriState) -> Command[Literal["supervisor"]]:
    result = insight_agent.invoke({"messages": state["messages"]})
    if "medical_flag" in str(result).lower() or "1200" in str(result):
        interrupt("Medical concern flagged. Awaiting human review.")
    return Command(
        update={"messages": result["messages"]},
        goto="supervisor",
    )


builder = StateGraph(NutriState)

builder.add_node("supervisor", supervisor_node)
builder.add_node("memory_agent", memory_node)
builder.add_node("nutrition_rag_agent", nutrition_node)
builder.add_node("planning_agent", planning_node)
builder.add_node("intake_agent", intake_node)
builder.add_node("insight_agent", insight_node)

builder.add_edge(START, "supervisor")

try:
    _checkpointer = get_checkpointer()
except Exception as e:
    _checkpointer = None
    print(f"NOTE: No PostgreSQL connection ({e}). Running without persistent memory.")

compiled = builder.compile(checkpointer=_checkpointer)
