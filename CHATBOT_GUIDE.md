# Financial RAG Chatbot - Quick Start Guide

## Overview
This implementation provides a production-ready RAG (Retrieval-Augmented Generation) chatbot for financial 10-K document analysis with LangSmith evaluation integration.

## Architecture

```
User Query
    ↓
Vector Store (FAISS) → Metadata Filtering → Top-K Retrieval
    ↓
LLM (OpenAI/Ollama) → Generation with Context
    ↓
LangSmith Logging → Evaluation & Analytics
    ↓
Response + Citations
```

## Quick Start

### 1. Setup Environment
```bash
# Install dependencies
pip install langchain langchain-openai langchain-community langsmith faiss-cpu sentence-transformers pandas

# Set environment variables
export LANGSMITH_API_KEY="your-api-key"
export LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
export OPENAI_API_KEY="your-openai-key"  # Optional, if using OpenAI
```

### 2. Running the Notebook
1. Open `scripts/chat.ipynb` in VS Code
2. Run cells sequentially:
   - Section 1: Load vector store and embeddings
   - Section 2: Initialize retriever
   - Section 3: Setup LLM (uncomment your preferred model)
   - Section 4: Create RAG chain
   - Section 5: Initialize chatbot
   - Section 6-9: Run examples or batch evaluations

### 3. Basic Usage
```python
# Single query
result = chatbot.chat("What are Apple's main risk factors?")
print(result['answer'])

# Save session
chatbot.save_session("session_2024.json")
```

## Key Components

### FinancialChatBot Class
Main chatbot interface with memory and LangSmith integration.

```python
chatbot = FinancialChatBot(rag_chain, client=client)
result = chatbot.chat(question, log_to_langsmith=True)
```

**Methods:**
- `chat(question)` - Get response with automatic logging
- `get_history()` - Get chat history as DataFrame
- `save_session(filepath)` - Persist chat session

### Advanced Retrieval (rag_utils.py)
```python
from rag_utils import AdvancedRetriever

retriever = AdvancedRetriever(vectorstore)

# Filter by company
docs = retriever.retrieve_with_company_filter(
    query="revenue trends", 
    company_filter=["AAPL", "MSFT"],
    k=5
)

# With reranking
docs = retriever.retrieve_with_reranking(
    query="risk assessment",
    company_preference={"AAPL": 0.1, "MSFT": 0.05}
)
```

### Evaluation Metrics
```python
from rag_utils import EvaluationMetrics

scores = EvaluationMetrics.composite_score(response, query)
# Returns:
# {
#   "relevance": 0.85,
#   "hallucination": 0.9,
#   "attribution": 0.8,
#   "length": 0.95,
#   "composite": 0.87
# }
```

### Session Management
```python
from rag_utils import SessionManager

session = SessionManager("eval_session")
session.add_interaction(query, response)
session.save_to_json("session.json")
print(session.get_stats())
```

## LangSmith Integration

### Automatic Logging
Every `chatbot.chat()` call automatically logs to LangSmith with:
- Input question
- Generated response
- Timestamp
- Custom tags

### Manual Evaluation Dataset
```python
from rag_utils import LangSmithEvaluator

evaluator = LangSmithEvaluator(client, project_name="financial-rag")

examples = [
    {"question": "What is Apple's revenue?", "expected_answer": "..."},
    {"question": "What are risk factors?", "expected_answer": "..."}
]

evaluator.create_evaluation_dataset(examples, "financial_qa_eval")
```

### View Results
1. Go to https://smith.langchain.com/
2. Find project "financial-rag-chat"
3. View runs, traces, and evaluation metrics
4. Compare LLM model performance over time

## Evaluation Metrics

### Relevance Score
Measures if response contains relevant financial information.
- Checks for financial keywords (revenue, risk, earnings, etc.)
- Evaluates query term coverage
- Score range: 0.0 - 1.0

### Hallucination Score
Detects if model admits knowledge limitations.
- Looks for "I don't have", "not available" phrases
- Higher score = better honesty
- Score range: 0.0 - 1.0

### Attribution Score
Checks for source citations.
- Looks for "[Source X]", "According to", etc.
- Score range: 0.0 - 1.0

### Length Score
Ensures response is appropriately detailed.
- Ideal range: 100-500 words
- Too short (<50 words): 0.2
- Too long (>1000 words): 0.4
- Score range: 0.0 - 1.0

### Composite Score
Weighted average of all metrics:
- Relevance: 40%
- Hallucination: 30%
- Attribution: 20%
- Length: 10%

## Configuration

### LLM Models

**OpenAI (Best Quality)**
```python
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
```
Requires: `OPENAI_API_KEY`

**Ollama (Local, Free)**
```python
llm = ChatOllama(model="mistral", temperature=0.3)
```
Requires: `ollama serve` running locally

**Ollama Models:**
- mistral (7B, balanced)
- llama2 (7B, general purpose)
- neural-chat (7B, instruction-tuned)
- orca-mini (3B, lightweight)

### Retriever Settings

**k (Number of documents)**
- k=3: Quick, focused responses
- k=5: Balanced (default)
- k=10: Comprehensive, may be verbose

**Search Type**
- "similarity": Default, uses cosine similarity
- "mmr": Maximal Marginal Relevance (more diverse)

### Metadata Filtering
```python
# Filter by company
retriever_typed = retriever.as_retriever(search_kwargs={
    "k": 5,
    "filter": {"company_name": "AAPL"}
})

# Available filter fields:
# - company_name: AAPL, GOOG, MSFT, NFLX, NVDA, UBER, etc.
# - section: risk_factors, management_discussion, business, etc.
```

## Batch Evaluation Example

```python
test_queries = [
    "What are the main risk factors?",
    "Compare revenue across companies",
    "What are cybersecurity risks?",
    "Summarize MD&A section",
    "List key business segments"
]

results = run_batch_evaluation(test_queries)
# Returns DataFrame with responses and metrics
```

## Troubleshooting

### "Vector store size is 0"
- Check path to embeddings: `../data/embeddings_E5_Large`
- Verify you ran `create_embeddings.ipynb` first

### LangSmith API errors
- Verify `LANGSMITH_API_KEY` is set
- Check endpoint: `https://api.smith.langchain.com`
- Test: `python -c "from langsmith import Client; print(Client())"`

### Slow responses
- Reduce k (number of retrieved documents)
- Use local LLM (Ollama) instead of OpenAI
- Ensure GPU is available for embeddings

### "Ollama not available"
- Run: `ollama serve`
- Download model: `ollama pull mistral`
- Verify: `curl http://localhost:11434/tags`

## Files & Structure

```
scripts/
├── chat.ipynb                 # Main chatbot notebook
├── rag_utils.py              # Utility functions
├── create_embeddings.ipynb   # (existing) Build FAISS index
└── ...other scripts

data/
├── embeddings_E5_Large/      # FAISS index (pre-built)
│   ├── index.faiss
│   └── index.pkl
├── chunked/                  # Document chunks
├── processed/                # Processed 10-K documents
└── raw/                      # Original HTML files
```

## Advanced Usage

### Custom Retrieval Chain
```python
from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

result = qa({"query": "Your question"})
print(result["result"])
print(result["source_documents"])
```

### Multi-Company Analysis
```python
companies = ["AAPL", "MSFT", "GOOG"]
for company in companies:
    docs = retriever.retrieve_with_company_filter(
        "revenue growth", 
        company_filter=[company],
        k=3
    )
    # Analyze per company
```

### Custom Evaluation
```python
from rag_utils import print_evaluation_report

response = "Apple reported revenue of $383B..."
scores = EvaluationMetrics.composite_score(response, "What is Apple's revenue?")
print_evaluation_report(scores)
```

## Performance Tips

1. **Batch queries** instead of individual calls
2. **Use company filters** to reduce search space
3. **Cache embeddings** if possible
4. **Run on GPU** for faster embeddings (set `device="cuda"`)
5. **Use local LLM** for specific use cases
6. **Monitor token usage** if using OpenAI

## Next Steps

1. Modify system prompt for specific use cases
2. Add custom evaluation metrics
3. Implement feedback loops for improvement
4. Deploy as API (Flask/FastAPI)
5. Build frontend UI for end users

## Support

For issues or improvements, check:
- LangSmith docs: https://docs.smith.langchain.com/
- LangChain docs: https://docs.langchain.com/
- FAISS: https://github.com/facebookresearch/faiss
