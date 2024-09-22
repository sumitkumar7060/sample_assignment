# sample_assignment

This is a generative task.  
Use OpenAI and Chroma DB.

In the `.env` file, set the following keys:

1. `AZURE_OPENAI_ENDPOINT`
2. `AZURE_OPENAI_API_KEY`

## Run the project

First, clone the repository, and then run:

```bash
streamlit run rag.py

chroma run --path "" --port 1998  for vector database