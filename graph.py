from langgraph.graph import StateGraph, END
from typing import TypedDict
from agents import supervisor_agent, web_search_agent, weather_agent, result_formatter


class AgentState(TypedDict, total=False):
    query: str
    route: str
    raw_result: str
    source: str
    final_answer: str


def route_decision(state: dict) -> str:
    return state["route"]


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_agent)
    graph.add_node("web_search", web_search_agent)
    graph.add_node("weather", weather_agent)
    graph.add_node("formatter", result_formatter)

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_decision,
        {"web_search": "web_search", "weather": "weather", "formatter": "formatter"},
    )

    graph.add_edge("web_search", "supervisor")
    graph.add_edge("weather", "supervisor")
    graph.add_edge("formatter", END)

    return graph.compile()


app_graph = build_graph()
