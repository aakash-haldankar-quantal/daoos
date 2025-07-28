import os
import json
import asyncio
import textwrap
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from langchain_openai import ChatOpenAI
# LangChain / LangGraph / MCP
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# ───────────────────────────────── Configuration ─────────────────────────
def _load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

_load_env()


OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
NOTION_MCP_TOKEN = os.getenv("NOTION_MCP_TOKEN")
NOTION_VERSION   = "2022-06-28"

if not CLAUDE_API_KEY or not NOTION_MCP_TOKEN:
    raise RuntimeError("Set CLAUDE_API_KEY and NOTION_MCP_TOKEN in env.")

notion_cfg = {
    "notion": {
        "command": "npx",
        "args": ["-y", "@notionhq/notion-mcp-server"],
        "transport": "stdio",
        "env": {
            "OPENAPI_MCP_HEADERS": json.dumps({
                "Authorization": f"Bearer {NOTION_MCP_TOKEN}",
                "Notion-Version": NOTION_VERSION,
            })
        },
    }
}

# ─────────────── Universal Workspace Knowledge ───────────────
@dataclass
class FieldConstraints:
    required: bool = False
    options: Optional[List[Dict[str, Any]]] = None

@dataclass
class PropertyDetails:
    name: str
    type: str
    constraints: FieldConstraints = field(default_factory=FieldConstraints)

@dataclass
class EntitySchema:
    id: str
    name: str
    properties: Dict[str, PropertyDetails] = field(default_factory=dict)

@dataclass
class WorkspaceEntity:
    id: str
    name: str

class UniversalWorkspaceKnowledge:
    def __init__(self):
        self.entities: Dict[str, WorkspaceEntity] = {}
        self.schemas: Dict[str, EntitySchema] = {}
        self.users: Dict[str, str] = {}       # name → id
        self.last_updated: datetime = datetime.now()

    def add_entity(self, eid: str, name: str):
        self.entities[eid] = WorkspaceEntity(eid, name)

    def add_schema(self, db_id: str, props: Dict[str, PropertyDetails]):
        self.schemas[db_id] = EntitySchema(db_id, self.entities[db_id].name, props)

    def add_user(self, uid: str, name: str):
        self.users[name] = uid

    def context(self) -> str:
        lines = ["=== WORKSPACE SCHEMA ==="]
        for eid, ent in self.entities.items():
            lines.append(f"\n*{ent.name}* (DB ID: {eid})")
            schema = self.schemas.get(eid)
            if schema:
                for pname, pdet in schema.properties.items():
                    req = " (required)" if pdet.constraints.required else ""
                    opts = ""
                    if pdet.constraints.options:
                        opts = " | opts: " + ", ".join(o["name"] for o in pdet.constraints.options)
                    lines.append(f"  • {pname}: {pdet.type}{req}{opts}")
        if self.users:
            lines.append("\n=== USERS ===")
            for name, uid in self.users.items():
                lines.append(f"  • {name} → {uid}")
        return "\n".join(lines)

# ────────────────────── Discovery via MCP ──────────────────────
async def discover_structure(node: ToolNode, know: UniversalWorkspaceKnowledge):
    # 1) Databases
    resp = await node.ainvoke({
        "messages":[AIMessage(content="", tool_calls=[{
            "name":"API-post-search",
            "args":{"filter":{"property":"object","value":"database"}},
            "id":"db_search"
        }])]
    })
    db_ids = []
    for m in resp["messages"]:
        if isinstance(m, ToolMessage):
            data = json.loads(m.content)
            for item in data.get("results", []):
                if item["object"]=="database":
                    eid = item["id"]
                    name = "".join(t["plain_text"] for t in item["title"])
                    know.add_entity(eid, name)
                    db_ids.append(eid)

    # 2) Schemas
    for db in db_ids:
        resp = await node.ainvoke({
            "messages":[AIMessage(content="", tool_calls=[{
                "name":"API-retrieve-a-database",
                "args":{"database_id":db},
                "id":"schema"
            }])]
        })
        for m in resp["messages"]:
            if isinstance(m, ToolMessage):
                data = json.loads(m.content)
                props = {}
                for pname, pdat in data["properties"].items():
                    typ = pdat["type"]
                    req = pname.lower() in ("name","title")
                    opts = None
                    if typ in ("select","status","multi_select"):
                        opts = pdat.get(typ,{}).get("options")
                    props[pname] = PropertyDetails(pname,typ,FieldConstraints(req,opts))
                know.add_schema(db, props)

    # 3) Users
    resp = await node.ainvoke({
        "messages":[AIMessage(content="", tool_calls=[{
            "name":"API-get-users","args":{},"id":"users"
        }])]
    })
    for m in resp["messages"]:
        if isinstance(m, ToolMessage):
            data = json.loads(m.content)
            for u in data.get("results",[]):
                know.add_user(u["id"], u.get("name","Unknown"))

    know.last_updated = datetime.now()
    return know

# ────────────────── System Prompt & Guardrails ──────────────────
GUARDRAILS = """
## MCP Usage Rules
- Always use `"parent": {"database_id": "<DB_ID>"}`
- For each property:
  - status → {"status": {"name": "..."}}
  - select → {"select": {"name": "..."}}
  - date   → {"date": {"start": "YYYY-MM-DD"}}
  - people → {"people": ["<USER_ID>", ...]}
"""

MISSING_FIELD_GUIDE = """
## When creating a new record:
1. Identify required fields from the schema.
2. If missing, ask:
   "The '<FieldName>' field is required. Would you like to provide it now?"
3. Validate user reply against type & options.
4. Repeat until all required fields are handled or user opts to skip.
"""

def make_system_prompt(know: UniversalWorkspaceKnowledge):
    return (
        "You are an interactive MCP‑driven Notion assistant.\n"
        "You know the full workspace schema and user list.\n"
        + GUARDRAILS
        + MISSING_FIELD_GUIDE
        + "\n" + know.context()
    )

# ──────────────────── Agent & Workflow ────────────────────
class State(MessagesState):
    pass

async def build_app():
    client = MultiServerMCPClient(notion_cfg)
    tools = await client.get_tools()
    
    # llm   = ChatAnthropic(model="claude-3-5-sonnet-20241022",
    #                       api_key=CLAUDE_API_KEY, temperature=0.1
    #                      ).bind_tools(tools)
    
    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=OPENAI_API_KEY,
        temperature=0.1,
    ).bind_tools(tools)
    
    node  = ToolNode(tools)

    know  = UniversalWorkspaceKnowledge()
    await discover_structure(node, know)

    async def agent_fn(state: State):
        prompt = make_system_prompt(know)
        sys = SystemMessage(content=prompt)
        hist = state["messages"]
        resp = await llm.ainvoke([sys] + hist)
        return {"messages": hist + [resp]}

    def router(state: State) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    wf = StateGraph(State)
    wf.add_node("agent", agent_fn)
    wf.add_node("tools", node)
    wf.add_edge(START, "agent")
    wf.add_conditional_edges("agent", router, {"tools":"tools", END:END})
    wf.add_edge("tools", "agent")
    return wf.compile(checkpointer=MemorySaver())

# ──────────────────────────── Main ────────────────────────────
async def main():
    app = await build_app()
    print("✨ Ready for MCP‑driven Notion interaction. Type 'quit' to exit.\n")
    config = {"configurable":{"thread_id":"mcp_session"}}

    while True:
        q = input("You: ").strip()
        if q.lower() in ("quit","exit"):
            break
        print("🤖 Assistant:", end=" ", flush=True)
        result = await app.ainvoke({"messages":[HumanMessage(content=q)]}, config=config)
        print(f"result: {result}")
        out = result["messages"][-1].content
        print(out, "\n")

if __name__ == "__main__":
    asyncio.run(main())
