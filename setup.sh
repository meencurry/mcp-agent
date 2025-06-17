# setup.sh - Setup script for the AI Agent Chat Interface

#!/bin/bash

echo "🚀 Setting up AI Agent Chat Interface..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing requirements..."
pip install chainlit==1.0.505
pip install langgraph==0.2.40
pip install langchain==0.3.7
pip install langchain-openai==0.2.8
pip install langchain-core==0.3.15
pip install chromadb==0.5.23
pip install python-dotenv==1.0.1
pip install pydantic==2.10.3
pip install aiohttp==3.11.10

# Create directories
echo "📁 Creating directories..."
mkdir -p chroma_db
mkdir -p safe_files
mkdir -p logs

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "🔑 Creating .env file..."
    cat > .env << EOL
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Chroma DB Configuration
CHROMA_PERSIST_DIRECTORY=./chroma_db

# Logging Configuration
LOG_LEVEL=INFO

# Chainlit Configuration
CHAINLIT_AUTH_SECRET=your_secret_key_here
EOL
    echo "⚠️ Please edit .env file and add your OpenAI API key!"
else
    echo "✅ .env file already exists"
fi

echo "✨ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file and add your OpenAI API key"
echo "2. Run: chainlit run app.py -w"
echo "3. Open http://localhost:8000 in your browser"
echo ""
echo "🛠️ Development mode:"
echo "chainlit run app.py -w --debug"