# RAG System for KCC Dataset

This project implements a Retrieval Augmented Generation (RAG) system for the KCC dataset using FastAPI backend and Streamlit frontend. The system uses Ollama LLM for text generation and FAISS for efficient vector storage and retrieval.

## Project Structure

```
.
├── src/
│   ├── api.py              # FastAPI backend implementation
│   ├── model.py            # LLM and RAG model implementation
│   ├── streamlit_app.py    # Streamlit frontend interface
│   ├── data_preprocessing.py # Data preprocessing utilities
│   ├── main.py            # Main application entry point
│   ├── vector_store.py    # Vector store management
│   └── __init__.py
├── data/                  # Directory for storing dataset and vectors
├── requirements.txt       # Project dependencies
└── README.md
```

## Features

- FastAPI backend for handling RAG operations
- Streamlit frontend for user interaction
- Efficient data preprocessing pipeline
- FAISS vector store for similarity search
- Integration with Ollama LLM
- Performance metrics tracking
- Data persistence for preprocessed data
- Error handling and system monitoring

## Technical Stack

- **Backend**: FastAPI
- **Frontend**: Streamlit
- **Vector Store**: FAISS
- **LLM**: Ollama
- **Data Processing**: Pandas, NumPy
- **Text Processing**: NLTK, Transformers

## Setup Instructions

1. Create a virtual environment:

   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:

   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the FastAPI backend:

   ```bash
   cd src
   uvicorn api:app --reload
   ```

5. Start the Streamlit frontend (in a new terminal):
   ```bash
   cd src
   streamlit run streamlit_app.py
   ```

![image](https://github.com/user-attachments/assets/cdc64256-4b4c-443e-82e5-1e3164598cb9)

## Usage

1. The system processes the KCC dataset (limited to 30,000 samples for memory efficiency)
2. Documents are preprocessed and stored in the vector store
3. Access the Streamlit interface at http://localhost:8501
4. Enter your query in the interface
5. The system will:
   - Retrieve relevant documents using RAG
   - Generate a response using the Ollama LLM
   - Display timing metrics for performance monitoring

## Performance Considerations

- Dataset is limited to 30,000 samples to manage memory usage
- Vector store persistence is implemented for faster subsequent runs
- Timing metrics are available for:
  - Document retrieval
  - Response generation
  - Total processing time

## Troubleshooting

1. If encountering "No documents" error:

   - Verify vector store initialization
   - Check data preprocessing completion

2. For environment-related issues:

   - Ensure virtual environment is activated
   - Verify all dependencies are installed
   - Run each server in a separate terminal

3. For performance issues:
   - Monitor timing metrics
   - Check system resource usage
   - Adjust batch sizes if needed

## Contributing

Feel free to submit issues and enhancement requests.

## License

[Specify your license here]
