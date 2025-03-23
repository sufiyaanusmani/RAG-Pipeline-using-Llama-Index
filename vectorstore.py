import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore


def get_vectorstore() -> ChromaVectorStore:
    """
    Initializes and returns a ChromaVectorStore instance.
    This function creates a persistent client for the Chroma database using the
    directory specified by Directory.VECTORSTORE_DIRECTORY. It then retrieves or
    creates a collection named "transcription_project" within the database and
    returns a ChromaVectorStore instance associated with this collection.

    Returns
    -------
        ChromaVectorStore: An instance of ChromaVectorStore associated with the
        "transcription_project" collection.

    """
    db = chromadb.PersistentClient(path=str("../the-server/vectorstore/"))
    chroma_collection = db.get_or_create_collection("transcription_project")
    return ChromaVectorStore(chroma_collection=chroma_collection)