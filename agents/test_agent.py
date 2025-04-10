from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()

class State(BaseModel):
    some_text: str = Field(description="Some text to be revised", default="hello world")

graph_builder = StateGraph(State)

def human_node(state: State):
    value = interrupt(
        # Any JSON serializable value to surface to the human.
        # For example, a question or a piece of text or a set of keys in the state
       {
          "text_to_revise": state.some_text
       }
    )
    # Update the state with the human's input or route the graph based on the input.
    return Command()

graph_builder.add_node('human_node', human_node)
graph_builder.add_edge(START, "human_node")


graph = graph_builder.compile(
    checkpointer=checkpointer # Required for `interrupt` to work
)

# # Run the graph until the interrupt
# thread_config = {"configurable": {"thread_id": "some_id"}}
# graph.invoke(some_input, config=thread_config)

# # Resume the graph with the human's input
# graph.invoke(Command(resume=value_from_human), config=thread_config)