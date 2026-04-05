"""
Utility functions for Financial RAG system with LangSmith evaluation.
"""

import json
from typing import List, Dict, Optional, Callable
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langsmith import Client
from langsmith.schemas import Run, Example
import pandas as pd


class AdvancedRetriever:
    """Advanced retrieval with metadata filtering and reranking."""
    
    def __init__(self, vectorstore: FAISS):
        self.vectorstore = vectorstore
        
    def retrieve_with_company_filter(
        self, 
        query: str, 
        company_filter: Optional[List[str]] = None,
        k: int = 5
    ) -> List[Document]:
        """Retrieve documents filtered by company."""
        # Get more results to filter
        docs = self.vectorstore.similarity_search(query, k=k*3)
        
        if company_filter:
            docs = [
                d for d in docs 
                if d.metadata.get("company_name") in company_filter
            ]
            return docs[:k]
        
        return docs[:k]
    
    def retrieve_with_section_filter(
        self, 
        query: str, 
        section_filter: Optional[List[str]] = None,
        k: int = 5
    ) -> List[Document]:
        """Retrieve documents filtered by section."""
        docs = self.vectorstore.similarity_search(query, k=k*3)
        
        if section_filter:
            docs = [
                d for d in docs 
                if d.metadata.get("section") in section_filter
            ]
            return docs[:k]
        
        return docs[:k]
    
    def retrieve_with_reranking(
        self, 
        query: str, 
        company_preference: Optional[Dict[str, float]] = None,
        k: int = 5
    ) -> List[tuple]:
        """Retrieve with company preference scoring."""
        docs = self.vectorstore.similarity_search_with_scores(query, k=k*2)
        
        reranked = []
        for doc, score in docs:
            boosted_score = score
            
            if company_preference:
                company = doc.metadata.get("company_name", "")
                if company in company_preference:
                    # Boost score for preferred companies
                    boost = company_preference[company]
                    boosted_score = score * (1 + boost)
            
            reranked.append((doc, score, boosted_score))
        
        # Sort by boosted score
        reranked.sort(key=lambda x: x[2], reverse=True)
        
        return reranked[:k]
    
    def get_metadata_stats(self) -> Dict[str, int]:
        """Get statistics about available metadata."""
        all_docs = self.vectorstore.similarity_search("", k=self.vectorstore.index.ntotal)
        
        companies = set()
        sections = set()
        
        for doc in all_docs:
            companies.add(doc.metadata.get("company_name", "Unknown"))
            sections.add(doc.metadata.get("section", "Unknown"))
        
        return {
            "total_documents": self.vectorstore.index.ntotal,
            "unique_companies": len(companies),
            "unique_sections": len(sections),
            "companies": list(companies),
            "sections": list(sections)
        }


class EvaluationMetrics:
    """Evaluation metrics for RAG system."""
    
    @staticmethod
    def relevance_score(response: str, query: str) -> float:
        """Score response relevance to query."""
        # Check for financial keywords
        financial_keywords = [
            "revenue", "income", "profit", "loss", "earnings",
            "cash", "asset", "liability", "equity", "risk",
            "operating", "margin", "growth", "year", "quarter"
        ]
        
        query_words = set(query.lower().split())
        response_lower = response.lower()
        
        matches = sum(1 for keyword in financial_keywords if keyword in response_lower)
        query_matches = sum(1 for word in query_words if word in response_lower)
        
        # Composite score
        keyword_score = min(matches / 3, 1.0)
        query_score = min(query_matches / len(query_words) if query_words else 0, 1.0)
        
        return (keyword_score * 0.6 + query_score * 0.4)
    
    @staticmethod
    def hallucination_score(response: str) -> float:
        """Score for hallucination detection (1.0 = no hallucination)."""
        hallucination_phrases = [
            "i don't have information",
            "i cannot find",
            "not available in the documents",
            "unclear from the context",
            "unable to determine"
        ]
        
        response_lower = response.lower()
        is_honest = any(phrase in response_lower for phrase in hallucination_phrases)
        
        # If admits limitation, score higher (good)
        return 0.9 if is_honest else 0.6
    
    @staticmethod
    def source_attribution_score(response: str) -> float:
        """Score for proper source attribution."""
        attribution_keywords = [
            "[source", "according to", "the document states",
            "as mentioned in", "from", "filing"
        ]
        
        response_lower = response.lower()
        has_attribution = any(keyword in response_lower for keyword in attribution_keywords)
        
        return 1.0 if has_attribution else 0.5
    
    @staticmethod
    def length_score(response: str, ideal_range: tuple = (100, 500)) -> float:
        """Score for response length appropriateness."""
        word_count = len(response.split())
        min_words, max_words = ideal_range
        
        if min_words <= word_count <= max_words:
            return 1.0
        elif word_count < min_words / 2:
            return 0.2
        elif word_count > max_words * 2:
            return 0.4
        else:
            return 0.7
    
    @staticmethod
    def composite_score(response: str, query: str) -> Dict[str, float]:
        """Calculate composite evaluation score."""
        scores = {
            "relevance": EvaluationMetrics.relevance_score(response, query),
            "hallucination": EvaluationMetrics.hallucination_score(response),
            "attribution": EvaluationMetrics.source_attribution_score(response),
            "length": EvaluationMetrics.length_score(response)
        }
        
        # Weighted average
        composite = (
            scores["relevance"] * 0.40 +
            scores["hallucination"] * 0.30 +
            scores["attribution"] * 0.20 +
            scores["length"] * 0.10
        )
        
        scores["composite"] = composite
        return scores


class SessionManager:
    """Manage chat sessions and persistence."""
    
    def __init__(self, session_name: Optional[str] = None):
        self.session_name = session_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.interactions = []
        
    def add_interaction(self, query: str, response: str, metadata: Dict = None):
        """Add interaction to session."""
        self.interactions.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "metadata": metadata or {}
        })
    
    def save_to_json(self, filepath: str):
        """Save session to JSON file."""
        session_data = {
            "session_name": self.session_name,
            "created_at": datetime.now().isoformat(),
            "interaction_count": len(self.interactions),
            "interactions": self.interactions
        }
        
        with open(filepath, "w") as f:
            json.dump(session_data, f, indent=2)
        
        print(f"✓ Session saved to {filepath}")
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert interactions to pandas DataFrame."""
        return pd.DataFrame(self.interactions)
    
    def get_stats(self) -> Dict:
        """Get session statistics."""
        df = self.to_dataframe()
        
        if df.empty:
            return {"interaction_count": 0}
        
        word_counts = df["response"].str.split().str.len()
        
        return {
            "interaction_count": len(self.interactions),
            "avg_response_length": word_counts.mean(),
            "min_response_length": word_counts.min(),
            "max_response_length": word_counts.max(),
            "session_duration": (
                pd.to_datetime(df["timestamp"].iloc[-1]) - 
                pd.to_datetime(df["timestamp"].iloc[0])
            ).total_seconds() if len(df) > 1 else 0
        }


class LangSmithEvaluator:
    """LangSmith evaluation integration."""
    
    def __init__(self, client: Client, project_name: str = "financial-rag"):
        self.client = client
        self.project_name = project_name
    
    def create_evaluation_dataset(
        self, 
        examples: List[Dict[str, str]],
        dataset_name: str
    ) -> str:
        """Create evaluation dataset in LangSmith."""
        try:
            dataset = self.client.create_dataset(
                dataset_name=dataset_name,
                description="Financial RAG evaluation examples"
            )
            
            for example in examples:
                self.client.create_example(
                    inputs={"question": example.get("question", "")},
                    outputs={"expected_answer": example.get("expected_answer", "")},
                    dataset_id=dataset.id
                )
            
            print(f"✓ Created dataset '{dataset_name}' with {len(examples)} examples")
            return dataset.id
            
        except Exception as e:
            print(f"⚠ Failed to create dataset: {e}")
            return None
    
    def log_chain_run(
        self, 
        query: str, 
        response: str,
        run_name: str = "financial_rag_query",
        tags: List[str] = None
    ):
        """Log a chain run to LangSmith."""
        try:
            self.client.create_run(
                name=run_name,
                inputs={"question": query},
                outputs={"answer": response},
                project_name=self.project_name,
                tags=tags or ["rag", "financial"]
            )
        except Exception as e:
            print(f"⚠ Failed to log run: {e}")


def format_retrieval_results(docs: List[Document], include_scores: bool = False) -> str:
    """Format retrieval results for display."""
    formatted = []
    
    for i, doc in enumerate(docs, 1):
        metadata = doc.metadata
        company = metadata.get("company_name", "Unknown")
        section = metadata.get("section", "Unknown").replace("_", " ").title()
        
        header = f"[Source {i}] {company} - {section}"
        content = doc.page_content
        
        formatted.append(f"{header}\n{content}")
    
    return "\n\n---\n\n".join(formatted)


def print_evaluation_report(scores: Dict[str, float]):
    """Print formatted evaluation report."""
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    
    for metric, score in scores.items():
        if metric != "composite":
            pct = score * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"{metric.upper():15} [{bar}] {pct:5.1f}%")
    
    print("-" * 60)
    composite = scores.get("composite", 0) * 100
    bar = "█" * int(composite / 5) + "░" * (20 - int(composite / 5))
    print(f"{'COMPOSITE':15} [{bar}] {composite:5.1f}%")
    print("=" * 60 + "\n")
