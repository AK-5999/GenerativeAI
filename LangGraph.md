# 🌐 What is LangGraph?
- LangGraph is a framework for building stateful, multi-agent AI workflows.
- It extends LangChain by introducing a graph-based execution model, where you define your workflow as a graph of nodes (agents, tools, functions) connected by edges (control flow).
- At its heart, LangGraph uses a state machine.
  - State: A dictionary-like object that stores information as the workflow progresses.
    - Each node updates the state, and the next step depends on both the state and defined edges.
    - This allows:
      - Memory persistence
      - Conditional execution (different paths depending on results)
      - Checkpoints (restart from a saved state)
     
- **LangGraph = a state-machine-driven workflow engine for LLMs, where nodes = execution units, edges = control flow, and state = persistent memory.**

## ⚡ Why LangGraph?
- Traditional LangChain chains are linear: Step 1 → Step 2 → Step 3. So, LangChain is great for chains (linear sequences of calls).
- But real-world workflows need branching, loops, memory, conditional execution.
- LangGraph gives this power by modeling workflows as graphs (nodes + edges + state managements).

<img width="1002" height="1024" alt="image" src="https://github.com/user-attachments/assets/60451741-42d1-40b7-95ea-16210a01919b" />

```
from langgraph.graph import StateGraph, END
from langgraph.nodes import ToolNode, LLMNode

# Define the graph 
graph = StateGraph()

# Add nodes
graph.add_node("search", ToolNode(tool=search_tool))
graph.add_node("summarize", LLMNode(llm=llm))

# Define edges (flow)
graph.add_edge("search", "summarize")
graph.add_edge("summarize", END)

# Compile the workflow
app = graph.compile()
```
Now app.invoke({"input": "Who is CEO of OpenAI?"}).
→ runs through the graph automatically.


## 🚀 Where LangGraph Shines
- Agents (LLMs that decide next steps)
- Multi-step reasoning (with loops/branches)
- Tool orchestration (deciding which tool to call)
- Memory and persistence (via graph state + checkpoints)
- Complex workflows (multi-agent conversations, RAG pipelines, evaluators)

## Execution: A deterministic but flexible execution flow,
- Start with an Initial State.
- The graph engine looks at the current node.
- The node executes → updates the state
- Based on edges + state, decide the next node.
- Repeat until you hit the END node.
<img width="1349" height="1412" alt="image" src="https://github.com/user-attachments/assets/ef01533d-6084-414b-aef8-77ead8a5a602" />

## Agents in LangGraph
1. What is an agent in LnagGraph?
  - Since LangGraph is general, agents are patterns you design, not fixed templates.
  - In LangGraph, an agent = a node (or set of nodes) that makes decisions dynamically
2. How Agents Work in Graphs
  - Agent reads state (user input, past results, memory).
  - Agent will look on the list of available actions/tools/nodes.
  - Based on prompt + context, Decides what action/tool/node should run next.
  - Updates state with its decision.
  - Graph execution continues based on that decision (via conditional edges).
  - Example Flow
    - Input: “What’s the weather in Delhi today?”
    - State so far: { "query": "weather in Delhi" }
    - Available actions: ["SearchTool", "Calculator", "AnswerDirectly"]
    - Agent LLM prompt → decides: “Use SearchTool”
    - Graph edge routes workflow to SearchTool node.
  - Control = Conditional Edges
    - Once LLM outputs its decision, LangGraph uses conditional edges to pick the next node.
    - That’s why agents in LangGraph feel more flexible:
    - You can define explicit rules (e.g., if state contains error=True, go to fallback).
    - Or let the LLM decide dynamically (free-form choice).
3. Types of Agents in LangGraph
  - Tool-Using Agent: Like LangChain’s ReAct agent, it reads state, chooses a tool, runs it, loops back.
    - “Which tool do I need?”
  - Planner Agent: Generates a plan (sequence of steps), then other nodes execute it.
    - “Step 1: Search, Step 2: Summarize, Step 3: Verify”
  - Critic/Evaluator Agent: Runs after another agent to evaluate or verify results.
    - “Output good enough? Yes/No”
  - Supervisor Agent: Acts like a “manager” node → looks at state + decides which agent to run next. Similar to a router or controller in workflow engines.
4. Why Agents Fit Well in LangGraph
  - Graph model naturally supports:
  - Dynamic decision-making (conditional edges).
  - Fallback paths (if one agent fails → switch to another).
  - Loops (Critic can send back work until it’s good).
  - Parallel execution (multiple agents run in parallel, results merged).
5. Key Difference from LangChain Agents
| Aspect      | LangChain Agents                | LangGraph Agents                                     |
| ----------- | ------------------------------- | ---------------------------------------------------- |
| Definition  | Predefined tool-using templates | Any decision-making node                             |
| Execution   | Loop until final answer         | Graph execution with branches & state                |
| Flexibility | Limited (ReAct, Conversational) | Unlimited (planner, critic, supervisor, multi-agent) |
| Memory      | Chat history                    | Full graph state with checkpoints                    |

 
## State Management in LangGraph
State in LangGraph is the evolving memory of the workflow — it carries data between nodes, enables branching, and allows checkpointing for recovery.
1. What is State?
  - State = shared memory that flows through the graph.
  - It is like a context object that each node can read, update, or add to.
  - Without state → nodes would be isolated and unaware of past steps.
2. How State Works?
  - Start with an initial state (e.g., user query).
  - Each node execution:
    - Reads the current state.
    - Produces output.
    - Updates the state with new info.
  - Updated state is passed to the next node.
3. Types of State Usage
  - Read-only: Node uses input but doesn’t modify state.
  - Update state: Node adds/modifies keys (e.g., adding “search_results”).
  - Conditional branching: Decisions based on state values.
4. Checkpoints
  - LangGraph allows checkpointing = saving state at specific steps.
  - If workflow crashes/fails → can resume from checkpoint instead of restarting.
  - Useful for long, expensive, multi-agent workflows.
5. Why State Matters
  - Enables memory across multiple steps.
  - Supports multi-agent collaboration (all agents work on the same state).
  - Makes workflows deterministic + debuggable.
  - Allows human-in-the-loop correction by resuming from saved state.

## Control Flow Features
LangGraph’s control flow = Fallback (reliability) + Multi-step decisions (structured reasoning) + Branching (flexibility), making workflows agentic and production-grade.
1. Fallback Mechanism
  - If one node fails (tool error, LLM timeout, bad output) → graph can route to a fallback node.
  - Ensures workflow doesn’t crash.
  - Example:
    - Try WebSearch → if fails → fallback to KnowledgeBase.

2. Multi-step Decision Making
  - Agents don’t need to decide everything in one shot.
  - You can model sequential reasoning steps in the graph.
  - Example:
    - Step 1: Decide if external data is needed.
    - Step 2: If yes, choose which tool.
    - Step 3: After tool, verify results.
  - Benefit: Transparent + controllable reasoning.

3. Branching Out
  - Graph can split into different paths depending on conditions.
  - Types of branching:
    - If–Else branching → Based on query type, route differently.
    - Parallel branching → Run multiple agents at the same time, merge results.
    - Looping branches → Feedback cycles (e.g., Summarizer → Critic → Summarizer until approved).

4. Why These Features Matter
  - Fallback → Reliability (production-ready).
  - Multi-step → Structured reasoning (traceable logic).
  - Branching → Flexibility (adapt to many query types).

5. Example Combined Flow
  - Imagine a research agent system:
  - Query: “Summarize latest AI news.”
    - Step 1 (Decision): Does this need external data? → Yes.
    - Step 2 (Branch): If data required → use WebSearch; else → go directly to LLM.
    - Step 3 (Fallback): If WebSearch fails → use KnowledgeBase.
    - Step 4 (Loop): Summarizer result → Critic Agent → if bad → loop back.
    - Step 5: Final summary → END.
   
## 🧠 LangGraph Memory & Checkpointing: Fault Tolerance for AI Agents

LangGraph’s memory + checkpointing = robust, fault-tolerant agents that can pause, resume, and recover seamlessly.

1. State Management: Memory in LangGraph
  - Stores state across agent steps.
  - Unlike LangChain “simple memory” (chat history only), LangGraph supports structured memory.
  - Example: It can remember
    - Previous tool outputs
    - Decisions made by agents
    - Partial workflow progress

2. Checkpointing: The "Save Point" System

  - **Checkpointing** is a core feature that provides robust fault tolerance. Think of it as **save points in a video game** 🎮.
  - **How it Works:** At every node or significant event in the workflow, LangGraph can **save the agent's complete state** (the checkpoint).
  - **The Benefit:** If the system crashes, an API fails, or a network times out, the workflow can **resume from the last successful checkpoint** instead of restarting from the beginning.
  - **Result:** Significant **cost savings** (fewer wasted LLM tokens) and increased **reliability**.

---

3. Event Sourcing Model: Transparency & Auditability

  - It stores every step—including the initial input, tool calls, tool outputs, and final decisions—as a chronological **log of events**.
  - This log makes the entire workflow **transparent and auditable**.
  - **Debug & Reproduce:** You can **replay** these event logs to debug failed runs or exactly reproduce a past execution.

---

4. Why It's Crucial for AI Agents
  - AI agents often execute **long, multi-step workflows** involving numerous tool calls, data processing, and complex reasoning (e.g., in research, planning, or data analysis).
  - **The Problem Without It (e.g., LangChain):** If an API call fails mid-way through a long sequence, the entire workflow is lost, requiring a full restart.
  - **The Solution With LangGraph:** Checkpointing ensures that even if one step fails (like an API timeout), the agent can **resume safely**, minimizing wasted time and expensive LLM calls.

5. Example Use Case: Data-Analysis Agent
  - Imagine a three-step data analysis agent:
      - Step1.  **Query database** (Success)
      - Step2.  **Run AI model** (System Crash/API Fail)
      - Step3.  **Summarize results**
      - If system crashes at Step 2,
        - With LangChain → everything restarts.
        - With LangGraph → resumes at Step 2 using checkpoint.
