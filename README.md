# Document (Textbook) QA Chatbot

A simple RAG-based chatbot made using LangChain, OpenAI embedding models, Cohere LLM, and ChromaDB. UI made with Gradio.
By default, the chatbot is for the "Alice in Wonderland" book.

## Running Locally
### With Gradio:
1. run `pip install -r requirements.txt`
3. run `python main.py`
4. go to `http://127.0.0.1:7860`

### On Console:
1. run `pip install -r requirements.txt`
2. in `main.py` comment `run_gradio_app()` and uncomment `take_console_input()`
3. run `python main.py`
