# simplified_app.py - A working version with error handling
import os
import asyncio
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
from typing_extensions import TypedDict

import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Agent State Definition
class AgentState(TypedDict):
    messages: List[BaseMessage]
    session_id: str
    current_step: str
    tool_results: Dict[str, Any]
    context: Dict[str, Any]
    user_input: str

# Initialize Chroma DB
class ChromaManager:
    def __init__(self, persist_directory: str = "./chroma_db"):
        try:
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            self.collection = self.client.get_or_create_collection(
                name="chat_history",
                metadata={"hnsw:space": "cosine"}
            )
            print(f"✅ ChromaDB initialized at {persist_directory}")
        except Exception as e:
            print(f"❌ ChromaDB initialization failed: {e}")
            # Fallback to in-memory storage
            self.client = chromadb.Client()
            self.collection = self.client.create_collection(
                name="chat_history",
                metadata={"hnsw:space": "cosine"}
            )
    
    def add_message(self, session_id: str, message: str, role: str, metadata: Dict = None):
        """Add a message to the vector database"""
        try:
            doc_id = f"{session_id}_{datetime.now().isoformat()}_{uuid.uuid4().hex[:8]}"
            self.collection.add(
                documents=[message],
                metadatas=[{
                    "session_id": session_id,
                    "role": role,
                    "timestamp": datetime.now().isoformat(),
                    **(metadata or {})
                }],
                ids=[doc_id]
            )
        except Exception as e:
            print(f"Error adding message to ChromaDB: {e}")
    
    def search_messages(self, query: str, session_id: str = None, n_results: int = 5):
        """Search for relevant messages"""
        try:
            where = {"session_id": session_id} if session_id else None
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where
            )
            return results
        except Exception as e:
            print(f"Error searching messages: {e}")
            return {"documents": [[]], "metadatas": [[]]}

# Simple MCP Tool Manager
class SimpleMCPManager:
    def __init__(self):
        self.tools = {}
        self._register_tools()
    
    def _register_tools(self):
        """Register simple tools"""
        self.tools = {
            "calculate": self._calculate,
            "search_knowledge": self._search_knowledge,
            "get_current_time": self._get_current_time
        }
    
    async def _calculate(self, expression: str):
        """Safe calculator"""
        try:
            # Only allow safe mathematical operations
            allowed_chars = set('0123456789+-*/.() ')
            if not all(c in allowed_chars for c in expression):
                return {"error": "Invalid characters in expression"}
            
            # Evaluate safely
            result = eval(expression, {"__builtins__": {}}, {})
            return {"result": result, "expression": expression}
        except Exception as e:
            return {"error": str(e)}
    
    async def _search_knowledge(self, query: str):
        """Search knowledge base"""
        chroma_manager = cl.user_session.get("chroma_manager")
        if chroma_manager:
            results = chroma_manager.search_messages(query, n_results=3)
            return {
                "results": results['documents'][0] if results['documents'] else [],
                "query": query
            }
        return {"results": [], "query": query}
    
    async def _get_current_time(self):
        """Get current time"""
        return {"current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    
    async def execute_tool(self, tool_name: str, **kwargs):
        """Execute a tool"""
        if tool_name in self.tools:
            return await self.tools[tool_name](**kwargs)
        else:
            return {"error": f"Tool {tool_name} not found"}
    
    def get_tool_descriptions(self):
        """Get tool descriptions"""
        return {
            "calculate": "Perform mathematical calculations",
            "search_knowledge": "Search conversation history",
            "get_current_time": "Get current date and time"
        }

# Simplified Agent
class SimpleAgent:
    def __init__(self, llm, mcp_manager: SimpleMCPManager, chroma_manager: ChromaManager):
        self.llm = llm
        self.mcp_manager = mcp_manager
        self.chroma_manager = chroma_manager
        self.graph = self._create_graph()
    
    def _create_graph(self):
        """Create the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("process", self._process_message)
        
        # Set entry point and end
        workflow.set_entry_point("process")
        workflow.add_edge("process", END)
        
        # Compile
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    async def _process_message(self, state: AgentState):
        """Process the user message"""
        user_input = state.get("user_input", "")
        session_id = state.get("session_id", "")
        
        # Create a step to show the thinking process
        async with cl.Step(name="🤔 Analyzing Input", type="run") as step:
            step.output = "Analyzing user input to determine required actions..."
            
            # Determine if we need to use tools
            action = self._determine_action(user_input)
            step.output = f"Detected action type: {action}"
        
        # Execute tool if needed
        tool_result = None
        if action != "chat":
            try:
                async with cl.Step(name=f"🛠️ Using {action.title()} Tool", type="run") as tool_step:
                    tool_step.output = f"Preparing to use {action} tool..."
                    
                    if action == "calculate":
                        # Extract numbers and operators
                        import re
                        expression = re.sub(r'[^0-9+\-*/.() ]', '', user_input)
                        if expression.strip():
                            tool_step.output = f"Calculating expression: {expression}"
                            tool_result = await self.mcp_manager.execute_tool("calculate", expression=expression.strip())
                    elif action == "search":
                        tool_step.output = f"Searching knowledge base for: {user_input}"
                        tool_result = await self.mcp_manager.execute_tool("search_knowledge", query=user_input)
                    elif action == "time":
                        tool_step.output = "Getting current time..."
                        tool_result = await self.mcp_manager.execute_tool("get_current_time")
                    
                    if tool_result:
                        if "error" in tool_result:
                            tool_step.output = f"❌ Tool error: {tool_result['error']}"
                        else:
                            tool_step.output = f"✅ Tool result: {tool_result}"
            except Exception as e:
                tool_result = {"error": str(e)}
                async with cl.Step(name="❌ Tool Error", type="run") as error_step:
                    error_step.output = f"Error executing tool: {str(e)}"
        
        # Generate response
        async with cl.Step(name="🧠 Generating Response", type="run") as response_step:
            response_step.output = "Processing tool results and generating response..."
            response = await self._generate_response(user_input, tool_result)
            response_step.output = "Response generated successfully!"
        
        # Create messages
        messages = [
            HumanMessage(content=user_input),
            AIMessage(content=response)
        ]
        
        # Save to database
        self.chroma_manager.add_message(session_id, user_input, "user")
        self.chroma_manager.add_message(session_id, response, "assistant")
        
        return {
            **state,
            "messages": messages,
            "current_step": "completed"
        }
    
    def _determine_action(self, user_input: str) -> str:
        """Determine what action to take"""
        user_input_lower = user_input.lower()
        
        if any(word in user_input_lower for word in ["calculate", "math", "compute", "+", "-", "*", "/"]):
            return "calculate"
        elif any(word in user_input_lower for word in ["search", "find", "look", "history"]):
            return "search"
        elif any(word in user_input_lower for word in ["time", "date", "when", "now"]):
            return "time"
        else:
            return "chat"
    
    async def _generate_response(self, user_input: str, tool_result: Dict = None) -> str:
        """Generate AI response"""
        try:
            # Build context
            context_parts = []
            if tool_result:
                if "error" in tool_result:
                    context_parts.append(f"Tool error: {tool_result['error']}")
                else:
                    context_parts.append(f"Tool result: {tool_result}")
            
            # Create messages
            messages = [
                SystemMessage(content="""You are a helpful AI assistant with access to tools for calculations, searching conversation history, and getting the current time. 
                Respond naturally and helpfully. If you used a tool, incorporate the results into your response."""),
            ]
            
            if context_parts:
                messages.append(SystemMessage(content="\n".join(context_parts)))
            
            messages.append(HumanMessage(content=user_input))
            
            # Get response from LLM
            response = await self.llm.ainvoke(messages)
            return response.content
            
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}"
    
    async def process_message(self, message: str, session_id: str) -> str:
        """Process a user message"""
        initial_state = {
            "messages": [],
            "session_id": session_id,
            "current_step": "start",
            "tool_results": {},
            "context": {},
            "user_input": message
        }
        
        config = {"configurable": {"thread_id": session_id}}
        
        try:
            result = await self.graph.ainvoke(initial_state, config)
            messages = result.get("messages", [])
            
            if messages and isinstance(messages[-1], AIMessage):
                return messages[-1].content
            else:
                return "I'm sorry, I couldn't generate a response."
                
        except Exception as e:
            print(f"Error processing message: {e}")
            return f"I apologize, but I encountered an error: {str(e)}"

# Initialize global components
try:
    chroma_manager = ChromaManager(os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db"))
    mcp_manager = SimpleMCPManager()
    
    # Check if OpenAI API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        print("⚠️ Warning: OpenAI API key not set. Please set OPENAI_API_KEY in your .env file")
        llm = None
    else:
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            api_key=api_key
        )
        print("✅ OpenAI LLM initialized")
        
except Exception as e:
    print(f"❌ Initialization error: {e}")
    chroma_manager = None
    mcp_manager = None
    llm = None

@cl.on_chat_start
async def start():
    """Initialize chat session"""
    if not llm:
        await cl.Message(
            content="❌ **Configuration Error**\n\nPlease set your OpenAI API key in the .env file and restart the application.",
            author="System"
        ).send()
        return
    
    session_id = str(uuid.uuid4())
    
    # Store session components
    cl.user_session.set("session_id", session_id)
    cl.user_session.set("chroma_manager", chroma_manager)
    cl.user_session.set("mcp_manager", mcp_manager)
    
    # Create agent
    try:
        agent = SimpleAgent(llm, mcp_manager, chroma_manager)
        cl.user_session.set("agent", agent)
        
        # Welcome message
        await cl.Message(
            content="""🤖 **AI Agent Chat Interface**

Welcome! I'm your AI assistant powered by:
- 🧠 **LangGraph** for intelligent workflow processing
- 🔧 **MCP Tools** for enhanced capabilities
- 📊 **ChromaDB** for conversation memory
- ⚡ **Chainlit** for this beautiful interface

**What I can do:**
- 🧮 **Calculate**: Ask me math questions like "What's 15 * 24?"
- 🔍 **Search**: Find information from our conversation history
- ⏰ **Time**: Get current date and time
- 💬 **Chat**: Have natural conversations about anything

Try asking me something like:
- "Calculate 123 + 456"
- "What time is it?"
- "Search for what we discussed about math"

What would you like to explore today?""",
            author="AI Agent"
        ).send()
        
    except Exception as e:
        await cl.Message(
            content=f"❌ Error initializing agent: {str(e)}",
            author="System"
        ).send()

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages"""
    try:
        # Get session components
        agent = cl.user_session.get("agent")
        session_id = cl.user_session.get("session_id")
        
        if not agent or not session_id:
            await cl.Message(
                content="❌ Session error. Please refresh the page.",
                author="System"
            ).send()
            return
        
        # Process message through agent
        response = await agent.process_message(message.content, session_id)
        
        # Send response
        await cl.Message(
            content=response,
            author="AI Agent"
        ).send()
        
    except Exception as e:
        error_msg = f"I apologize, but I encountered an error: {str(e)}"
        await cl.Message(
            content=error_msg,
            author="AI Agent"
        ).send()
        print(f"Error in main handler: {e}")

@cl.on_chat_end
async def end():
    """Handle chat end"""
    session_id = cl.user_session.get("session_id")
    if session_id:
        print(f"Chat session {session_id[:8]}... ended")

# Custom CSS for better UI
@cl.on_settings_update
async def setup_agent(settings):
    print("Settings updated:", settings)

# Health check endpoint for monitoring
async def health_check():
    """Health check for the application"""
    status = {
        "status": "healthy",
        "components": {
            "chroma_db": chroma_manager is not None,
            "mcp_tools": mcp_manager is not None,
            "llm": llm is not None
        },
        "timestamp": datetime.now().isoformat()
    }
    return status

if __name__ == "__main__":
    print("🚀 Starting AI Agent Chat Interface...")
    print("📋 Make sure to:")
    print("   1. Set OPENAI_API_KEY in .env file")
    print("   2. Run: chainlit run simplified_app.py -w")
    print("   3. Open http://localhost:8000")
    
    # Print component status
    print(f"\n📊 Component Status:")
    print(f"   ChromaDB: {'✅' if chroma_manager else '❌'}")
    print(f"   MCP Tools: {'✅' if mcp_manager else '❌'}")
    print(f"   OpenAI LLM: {'✅' if llm else '❌'}")
    
    if not llm:
        print("\n⚠️  Warning: OpenAI API key not configured!")
        print("   Please add OPENAI_API_KEY to your .env file")
