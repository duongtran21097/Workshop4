"""
Langchain RAG chain for the Meeting AI Assistant chat tab.

Architecture:
  VectorStoreRetriever — wraps vector_store query function
  build_rag_chain      — returns a ConversationalRetrievalChain
                         (memory is managed externally by the caller)
"""

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_openai import ChatOpenAI
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from pydantic import ConfigDict

from .vector_store import query_relevant_documents

ROLE_PROMPTS = {
    "Manager": "Focus on decisions, risks, priorities, and high-level outcomes.",
    "Developer": "Focus on implementation details, blockers, dependencies, and tasks.",
    "QA": "Focus on test coverage, edge cases, validation, and quality risks.",
}


class VectorStoreRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    n_results: int = 3

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        results = query_relevant_documents(query, n_results=self.n_results)
        return [Document(page_content=r["page_content"], metadata=r["metadata"]) for r in results]


def build_rag_chain(
    base_url: str, api_key: str, role: str = "Manager"
) -> ConversationalRetrievalChain:
    """
    Build a stateless ConversationalRetrievalChain.
    Chat history is passed in via chain.invoke({"question": ..., "chat_history": [...]}).
    """
    llm = ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model="GPT-4o",
        temperature=0.4,
    )

    prompt = PromptTemplate(
        template=(
            "You are a helpful meeting assistant. "
            f"{ROLE_PROMPTS.get(role, '')}\n\n"
            "Use the retrieved meeting notes below to answer accurately. "
            "Answer in the same language as the question.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n"
            "Answer:"
        ),
        input_variables=["context", "question"],
    )

    # Condense prompt that respects the original question's language
    condense_prompt = PromptTemplate(
        template=(
            "Given the conversation history and a follow-up question, rephrase "
            "the follow-up as a standalone question. Keep it in the SAME LANGUAGE "
            "as the follow-up question.\n\n"
            "Chat History:\n{chat_history}\n\n"
            "Follow-up: {question}\n"
            "Standalone question:"
        ),
        input_variables=["chat_history", "question"],
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=VectorStoreRetriever(),
        condense_question_prompt=condense_prompt,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=False,
        verbose=False,
    )