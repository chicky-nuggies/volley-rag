# Things I Learned Building Volley RAG

## LangChain tool artifacts: a second output channel

LangChain tools do not have to return only text for the LLM. With `response_format="content_and_artifact"`, a tool returns `(content, artifact)`: `content` is shown to the model, while `artifact` is attached to the resulting `ToolMessage` for application code to use. In this chatbot, the retrieval tool returns readable context to the LLM and structured source records to `/chat` from the same search—an interesting pattern for citations, IDs, metadata, and other data the model does not need to see.
