# mcp_integration.py - Complete MCP Tools Integration

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import inspect
from enum import Enum

# MCP Protocol Types
class MCPResourceType(Enum):
    TEXT = "text"
    JSON = "json"
    BINARY = "binary"
    IMAGE = "image"

class MCPToolType(Enum):
    FUNCTION = "function"
    RESOURCE = "resource"
    PROMPT = "prompt"

@dataclass
class MCPParameter:
    """MCP Tool Parameter Definition"""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    enum: List[str] = field(default_factory=list)
    
    def to_schema(self) -> Dict[str, Any]:
        schema = {
            "type": self.type,
            "description": self.description
        }
        if self.enum:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
        return schema

@dataclass
class MCPToolSchema:
    """MCP Tool Schema Definition"""
    name: str
    description: str
    parameters: List[MCPParameter]
    returns: Dict[str, Any] = field(default_factory=dict)
    tool_type: MCPToolType = MCPToolType.FUNCTION
    
    def to_openai_function_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format"""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = param.to_schema()
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }

class MCPTool(ABC):
    """Abstract Base Class for MCP Tools"""
    
    def __init__(self, schema: MCPToolSchema):
        self.schema = schema
        self.execution_count = 0
        self.last_executed = None
        self.success_rate = 1.0
        
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with given parameters"""
        pass
    
    def validate_parameters(self, **kwargs) -> bool:
        """Validate input parameters against schema"""
        for param in self.schema.parameters:
            if param.required and param.name not in kwargs:
                raise ValueError(f"Missing required parameter: {param.name}")
            
            if param.name in kwargs:
                value = kwargs[param.name]
                # Type validation
                if param.type == "string" and not isinstance(value, str):
                    raise TypeError(f"Parameter {param.name} must be string")
                elif param.type == "integer" and not isinstance(value, int):
                    raise TypeError(f"Parameter {param.name} must be integer")
                elif param.type == "number" and not isinstance(value, (int, float)):
                    raise TypeError(f"Parameter {param.name} must be number")
                elif param.type == "boolean" and not isinstance(value, bool):
                    raise TypeError(f"Parameter {param.name} must be boolean")
                
                # Enum validation
                if param.enum and value not in param.enum:
                    raise ValueError(f"Parameter {param.name} must be one of: {param.enum}")
        
        return True

class MCPToolRegistry:
    """Registry for managing MCP Tools"""
    
    def __init__(self):
        self.tools: Dict[str, MCPTool] = {}
        self.logger = logging.getLogger(__name__)
    
    def register(self, tool: MCPTool):
        """Register a new MCP tool"""
        self.tools[tool.schema.name] = tool
        self.logger.info(f"Registered MCP tool: {tool.schema.name}")
    
    def unregister(self, tool_name: str):
        """Unregister an MCP tool"""
        if tool_name in self.tools:
            del self.tools[tool_name]
            self.logger.info(f"Unregistered MCP tool: {tool_name}")
    
    def get_tool(self, tool_name: str) -> Optional[MCPTool]:
        """Get a tool by name"""
        return self.tools.get(tool_name)
    
    def list_tools(self) -> List[str]:
        """List all registered tool names"""
        return list(self.tools.keys())
    
    def get_schemas(self) -> List[Dict[str, Any]]:
        """Get all tool schemas in OpenAI function format"""
        return [tool.schema.to_openai_function_schema() for tool in self.tools.values()]
    
    async def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool with error handling and metrics"""
        if tool_name not in self.tools:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found",
                "available_tools": self.list_tools()
            }
        
        tool = self.tools[tool_name]
        start_time = datetime.now()
        
        try:
            # Validate parameters
            tool.validate_parameters(**kwargs)
            
            # Execute tool
            result = await tool.execute(**kwargs)
            
            # Update metrics
            tool.execution_count += 1
            tool.last_executed = datetime.now()
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "result": result,
                "tool": tool_name,
                "execution_time": execution_time,
                "metadata": {
                    "execution_count": tool.execution_count,
                    "last_executed": tool.last_executed.isoformat()
                }
            }
            
        except Exception as e:
            # Update failure metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            tool.success_rate = (tool.success_rate * tool.execution_count) / (tool.execution_count + 1)
            tool.execution_count += 1
            
            self.logger.error(f"Tool execution failed for {tool_name}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "tool": tool_name,
                "execution_time": execution_time
            }

# Concrete MCP Tool Implementations

class CalculatorTool(MCPTool):
    """Advanced calculator with mathematical functions"""
    
    def __init__(self):
        schema = MCPToolSchema(
            name="calculator",
            description="Perform mathematical calculations with support for basic and advanced operations",
            parameters=[
                MCPParameter(
                    name="expression",
                    type="string",
                    description="Mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)', 'sin(3.14/2)')"
                ),
                MCPParameter(
                    name="precision",
                    type="integer",
                    description="Number of decimal places for the result",
                    required=False,
                    default=2
                )
            ]
        )
        super().__init__(schema)
    
    async def execute(self, expression: str, precision: int = 2) -> Dict[str, Any]:
        """Execute mathematical calculation"""
        import math
        import re
        
        # Security: Only allow safe mathematical operations
        allowed_names = {
            # Basic operations are handled by eval
            # Math functions
            "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
            "pow": pow, "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
            "tan": math.tan, "log": math.log, "log10": math.log10,
            "exp": math.exp, "ceil": math.ceil, "floor": math.floor,
            # Constants
            "pi": math.pi, "e": math.e
        }
        
        # Remove any potentially dangerous characters/functions
        dangerous_patterns = [
            r'__\w+__', r'import\s+', r'exec\s*\(', r'eval\s*\(',
            r'open\s*\(', r'file\s*\(', r'input\s*\(', r'raw_input\s*\('
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, expression, re.IGNORECASE):
                return {"error": f"Expression contains forbidden operations: {pattern}"}
        
        try:
            # Evaluate the expression
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            
            # Round to specified precision
            if isinstance(result, (int, float)):
                result = round(result, precision)
            
            return {
                "result": result,
                "expression": expression,
                "precision": precision,
                "type": type(result).__name__
            }
            
        except Exception as e:
            return {"error": f"Calculation error: {str(e)}"}

class WebSearchTool(MCPTool):
    """Web search tool using DuckDuckGo API"""
    
    def __init__(self):
        schema = MCPToolSchema(
            name="web_search",
            description="Search the web for information using DuckDuckGo",
            parameters=[
                MCPParameter(
                    name="query",
                    type="string",
                    description="Search query"
                ),
                MCPParameter(
                    name="max_results",
                    type="integer",
                    description="Maximum number of results to return",
                    required=False,
                    default=5
                ),
                MCPParameter(
                    name="safe_search",
                    type="string",
                    description="Safe search setting",
                    required=False,
                    default="moderate",
                    enum=["strict", "moderate", "off"]
                )
            ]
        )
        super().__init__(schema)
    
    async def execute(self, query: str, max_results: int = 5, safe_search: str = "moderate") -> Dict[str, Any]:
        """Execute web search"""
        import aiohttp
        
        try:
            async with aiohttp.ClientSession() as session:
                # DuckDuckGo Instant Answer API
                url = "https://api.duckduckgo.com/"
                params = {
                    "q": query,
                    "format": "json",
                    "no_html": "1",
                    "skip_disambig": "1",
                    "safe_search": safe_search
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Extract relevant information
                        results = []
                        
                        # Abstract answer
                        if data.get("Abstract"):
                            results.append({
                                "type": "abstract",
                                "text": data["Abstract"],
                                "source": data.get("AbstractSource", "DuckDuckGo")
                            })
                        
                        # Related topics
                        for topic in data.get("RelatedTopics", [])[:max_results-1]:
                            if isinstance(topic, dict) and "Text" in topic:
                                results.append({
                                    "type": "related",
                                    "text": topic["Text"],
                                    "url": topic.get("FirstURL", "")
                                })
                        
                        return {
                            "query": query,
                            "results": results[:max_results],
                            "total_found": len(results),
                            "safe_search": safe_search
                        }
                    else:
                        return {"error": f"Search API returned status {response.status}"}
                        
        except Exception as e:
            return {"error": f"Web search failed: {str(e)}"}

class FileManagerTool(MCPTool):
    """Safe file management tool"""
    
    def __init__(self, safe_directory: str = "./safe_files"):
        schema = MCPToolSchema(
            name="file_manager",
            description="Manage files in a safe directory",
            parameters=[
                MCPParameter(
                    name="action",
                    type="string",
                    description="Action to perform",
                    enum=["list", "read", "write", "delete", "exists"]
                ),
                MCPParameter(
                    name="filename",
                    type="string",
                    description="Name of the file",
                    required=False
                ),
                MCPParameter(
                    name="content",
                    type="string",
                    description="Content to write to file",
                    required=False
                )
            ]
        )
        super().__init__(schema)
        self.safe_directory = safe_directory
        
        # Ensure safe directory exists
        import os
        os.makedirs(safe_directory, exist_ok=True)
    
    async def execute(self, action: str, filename: str = None, content: str = None) -> Dict[str, Any]:
        """Execute file operation"""
        import os
        from pathlib import Path
        
        safe_path = Path(self.safe_directory)
        
        try:
            if action == "list":
                files = [f.name for f in safe_path.glob("*") if f.is_file()]
                return {
                    "action": "list",
                    "files": files,
                    "count": len(files)
                }
            
            elif action == "exists" and filename:
                file_path = safe_path / filename
                exists = file_path.exists() and file_path.is_file()
                return {
                    "action": "exists",
                    "filename": filename,
                    "exists": exists
                }
            
            elif action == "read" and filename:
                file_path = safe_path / filename
                if not file_path.exists():
                    return {"error": f"File '{filename}' not found"}
                
                with open(file_path, "r", encoding="utf-8") as f:
                    file_content = f.read()
                
                return {
                    "action": "read",
                    "filename": filename,
                    "content": file_content,
                    "size": len(file_content)
                }
            
            elif action == "write" and filename and content is not None:
                file_path = safe_path / filename
                
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                
                return {
                    "action": "write",
                    "filename": filename,
                    "size": len(content),
                    "success": True
                }
            
            elif action == "delete" and filename:
                file_path = safe_path / filename
                if not file_path.exists():
                    return {"error": f"File '{filename}' not found"}
                
                file_path.unlink()
                return {
                    "action": "delete",
                    "filename": filename,
                    "success": True
                }
            
            else:
                return {"error": f"Invalid action '{action}' or missing required parameters"}
                
        except Exception as e:
            return {"error": f"File operation failed: {str(e)}"}

class KnowledgeSearchTool(MCPTool):
    """Search conversation history and knowledge base"""
    
    def __init__(self, chroma_manager):
        schema = MCPToolSchema(
            name="knowledge_search",
            description="Search conversation history and knowledge base using semantic search",
            parameters=[
                MCPParameter(
                    name="query",
                    type="string",
                    description="Search query"
                ),
                MCPParameter(
                    name="max_results",
                    type="integer",
                    description="Maximum number of results",
                    required=False,
                    default=5
                ),
                MCPParameter(
                    name="session_id",
                    type="string",
                    description="Limit search to specific session",
                    required=False
                )
            ]
        )
        super().__init__(schema)
        self.chroma_manager = chroma_manager
    
    async def execute(self, query: str, max_results: int = 5, session_id: str = None) -> Dict[str, Any]:
        """Execute knowledge search"""
        try:
            results = self.chroma_manager.search_messages(
                query=query,
                session_id=session_id,
                n_results=max_results
            )
            
            # Format results
            formatted_results = []
            if results.get("documents") and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                    formatted_results.append({
                        "content": doc,
                        "metadata": metadata,
                        "relevance_score": results["distances"][0][i] if results.get("distances") else None
                    })
            
            return {
                "query": query,
                "results": formatted_results,
                "total_found": len(formatted_results),
                "session_filter": session_id
            }
            
        except Exception as e:
            return {"error": f"Knowledge search failed: {str(e)}"}

# MCP Integration with LangGraph Agent

class MCPIntegratedAgent:
    """Agent with full MCP tool integration"""
    
    def __init__(self, llm, chroma_manager):
        self.llm = llm
        self.chroma_manager = chroma_manager
        self.mcp_registry = MCPToolRegistry()
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register default MCP tools"""
        # Register built-in tools
        self.mcp_registry.register(CalculatorTool())
        self.mcp_registry.register(WebSearchTool())
        self.mcp_registry.register(FileManagerTool())
        self.mcp_registry.register(KnowledgeSearchTool(self.chroma_manager))
    
    def register_custom_tool(self, tool: MCPTool):
        """Register a custom MCP tool"""
        self.mcp_registry.register(tool)
    
    async def determine_tool_usage(self, user_input: str) -> List[Dict[str, Any]]:
        """Use LLM to determine which tools to use"""
        system_prompt = f"""
        You are a tool selection assistant. Given a user input, determine which tools should be used.
        
        Available tools:
        {json.dumps(self.mcp_registry.get_schemas(), indent=2)}
        
        Respond with a JSON array of tool calls in this format:
        [
            {{"tool": "tool_name", "parameters": {{"param1": "value1"}}}},
            ...
        ]
        
        If no tools are needed, respond with: []
        
        User input: {user_input}
        """
        
        try:
            response = await self.llm.ainvoke([{"role": "system", "content": system_prompt}])
            tool_calls = json.loads(response.content.strip())
            return tool_calls if isinstance(tool_calls, list) else []
        except Exception as e:
            print(f"Error determining tool usage: {e}")
            return []
    
    async def process_message_with_tools(self, user_input: str, session_id: str) -> str:
        """Process message with automatic tool selection and execution"""
        # Determine which tools to use
        tool_calls = await self.determine_tool_usage(user_input)
        
        # Execute tools
        tool_results = []
        for tool_call in tool_calls:
            tool_name = tool_call.get("tool")
            parameters = tool_call.get("parameters", {})
            
            if tool_name:
                result = await self.mcp_registry.execute_tool(tool_name, **parameters)
                tool_results.append({
                    "tool": tool_name,
                    "parameters": parameters,
                    "result": result
                })
        
        # Generate response using LLM with tool results
        context = ""
        if tool_results:
            context = f"Tool execution results:\n{json.dumps(tool_results, indent=2)}\n\n"
        
        system_prompt = f"""
        You are a helpful AI assistant. Use the tool results (if any) to provide a comprehensive response.
        
        {context}User message: {user_input}
        
        Provide a natural, helpful response that incorporates the tool results when relevant.
        """
        
        try:
            response = await self.llm.ainvoke([{"role": "system", "content": system_prompt}])
            return response.content
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}"

# Usage Example and Integration

async def example_usage():
    """Example of how to use the MCP integrated agent"""
    from langchain_openai import ChatOpenAI
    
    # Initialize components
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    # Assume chroma_manager is initialized
    
    # Create MCP integrated agent
    agent = MCPIntegratedAgent(llm, None)  # Pass actual chroma_manager
    
    # Example interactions
    test_messages = [
        "Calculate the square root of 144",
        "Search the web for latest AI news",
        "Create a file called 'notes.txt' with the content 'Hello World'",
        "What did we discuss about mathematics earlier?"
    ]
    
    for message in test_messages:
        print(f"User: {message}")
        response = await agent.process_message_with_tools(message, "test_session")
        print(f"Agent: {response}\n")

if __name__ == "__main__":
    asyncio.run(example_usage())
