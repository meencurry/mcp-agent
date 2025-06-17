# AI Agent Chat Interface

A sophisticated AI chat interface built with **Chainlit**, **LangGraph**, **MCP Tools**, and **ChromaDB** for intelligent conversation management and tool integration.

## 🌟 Features

- **🧠 Intelligent Agent**: Powered by LangGraph for complex workflow processing
- **🔧 MCP Tool Integration**: Extensible tool system for calculations, search, and more
- **📊 Vector Memory**: ChromaDB for semantic conversation history
- **⚡ Real-time Interface**: Beautiful Chainlit web interface
- **🛡️ Error Handling**: Robust error handling and fallback mechanisms
- **📈 Usage Analytics**: Tool usage statistics and performance monitoring

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Chainlit UI   │◄───┤   LangGraph     │◄───┤   MCP Tools     │
│                 │    │   Agent         │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Session       │    │   ChromaDB      │    │   OpenAI LLM    │
│   Management    │    │   Vector Store  │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key
- Git

### Installation

1. **Clone the repository**:
```bash
git clone <your-repo>
cd ai-agent-chat
```

2. **Run the setup script**:
```bash
chmod +x setup.sh
./setup.sh
```

3. **Configure environment**:
Edit `.env` file and add your OpenAI API key:
```env
OPENAI_API_KEY=sk-your-actual-openai-api-key-here
```

4. **Start the application**:
```bash
chainlit run simplified_app.py -w
```

5. **Open in browser**:
Navigate to `http://localhost:8000`

## 🔧 Manual Installation

If the setup script doesn't work, follow these manual steps:

### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install chainlit==1.0.505
pip install langgraph==0.2.40
pip install langchain==0.3.7
pip install langchain-openai==0.2.8
pip install langchain-core==0.3.15
pip install chromadb==0.5.23
pip install python-dotenv==1.0.1
pip install pydantic==2.10.3
pip install aiohttp==3.11.10
```

### 3. Create Directory Structure
```bash
mkdir -p chroma_db safe_files logs
```

### 4. Create Configuration File
Create `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key_here
CHROMA_PERSIST_DIRECTORY=./chroma_db
LOG_LEVEL=INFO
CHAINLIT_AUTH_SECRET=your_secret_key_here
```

## 📁 Project Structure

```
ai-agent-chat/
├── simplified_app.py          # Main application file
├── app.py                     # Full-featured version
├── mcp_tools.py              # Advanced MCP tool implementations
├── setup.sh                  # Automated setup script
├── requirements.txt          # Python dependencies
├── .env                      # Environment configuration
├── README.md                 # This file
├── chroma_db/               # ChromaDB persistence directory
├── safe_files/              # Safe file storage for tools
└── logs/                    # Application logs
```

## 🛠️ Available Tools

### 1. Calculator Tool
- **Function**: Perform mathematical calculations
- **Usage**: "Calculate 15 * 24 + 100"
- **Features**: Safe expression evaluation

### 2. Knowledge Search Tool
- **Function**: Search conversation history
- **Usage**: "Search for what we discussed about math"
- **Features**: Semantic search using vector embeddings

### 3. Time Tool
- **Function**: Get current date and time
- **Usage**: "What time is it?"
- **Features**: Formatted timestamp

## 🔌 Adding New Tools

To add a new MCP tool, follow this pattern:

```python
# In SimpleMCPManager class
async def _your_new_tool(self, param1: str, param2: int = 10):
    """Your tool description"""
    try:
        # Tool implementation
        result = process_something(param1, param2)
        return {"result": result, "status": "success"}
    except Exception as e:
        return {"error": str(e)}

# Register the tool
def _register_tools(self):
    self.tools = {
        # ... existing tools
        "your_new_tool": self._your_new_tool,
    }
```

## 🎨 Customization

### UI Customization
Modify the welcome message and styling in the `@cl.on_chat_start` function:

```python
await cl.Message(
    content="Your custom welcome message",
    author="Custom Agent"
).send()
```

### Agent Behavior
Customize the agent's decision logic in `_determine_action()`:

```python
def _determine_action(self, user_input: str) -> str:
    # Add your custom logic here
    if "your_keyword" in user_input.lower():
        return "your_action"
    # ... rest of logic
```

## 📊 Monitoring & Analytics

The application includes built-in monitoring:

- **Tool Usage Statistics**: Track which tools are used most
- **Performance Metrics**: Monitor response times
- **Error Tracking**: Log and handle errors gracefully
- **Session Management**: Track user sessions

## 🐛 Troubleshooting

### Common Issues

1. **"OpenAI API key not set" error**:
   - Check your `.env` file
   - Ensure the key starts with `sk-`
   - Restart the application

2. **ChromaDB initialization fails**:
   - Check write permissions in the project directory
   - Ensure `chroma_db` directory exists
   - Try deleting `chroma_db` folder and restarting

3. **Module import errors**:
   - Activate your virtual environment
   - Reinstall dependencies: `pip install -r requirements.txt`

4. **Chainlit won't start**:
   - Check if port 8000 is available
   - Try: `chainlit run simplified_app.py -w --port 8001`

### Debug Mode

Run in debug mode for detailed logs:
```bash
chainlit run simplified_app.py -w --debug
```

### Logs

Check application logs:
```bash
tail -f logs/app.log
```

## 🚀 Deployment

### Local Development
```bash
chainlit run simplified_app.py -w
```

### Production Deployment

#### Using Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["chainlit", "run", "simplified_app.py", "--host", "0.0.0.0", "--port", "8000"]
```

#### Using Gunicorn
```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker simplified_app:app --bind 0.0.0.0:8000
```

### Environment Variables for Production
```env
OPENAI_API_KEY=your_production_api_key
CHROMA_PERSIST_DIRECTORY=/app/data/chroma_db
LOG_LEVEL=WARNING
CHAINLIT_AUTH_SECRET=your_secure_secret_key
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests if applicable
5. Commit: `git commit -m "Add feature"`
6. Push: `git push origin feature-name`
7. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Chainlit** - For the amazing chat interface framework
- **LangGraph** - For powerful agent workflow management
- **ChromaDB** - For vector storage and semantic search
- **OpenAI** - For the language model capabilities

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Search existing issues on GitHub
3. Create a new issue with:
   - Your Python version
   - Error messages
   - Steps to reproduce

## 🔮 Roadmap

- [ ] Add more MCP tools (web search, file operations)
- [ ] Implement user authentication
- [ ] Add conversation export functionality
- [ ] Create admin dashboard
- [ ] Add support for multiple LLM providers
- [ ] Implement tool marketplace
- [ ] Add voice interface support

---

**Built with ❤️ using Chainlit, LangGraph, and ChromaDB**