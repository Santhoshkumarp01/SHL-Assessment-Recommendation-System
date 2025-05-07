from rag import load_assessments, build_vector_store

documents = load_assessments("assessments.csv")
build_vector_store(documents, save_path="vector_store.faiss")