# Aera RAG

The retrieval-augmented generation service behind the movie assistant in [Aera Movies](https://github.com/wxqryy/AeraMovies). It rewrites a natural-language request into a search query, retrieves relevant films from a FAISS index, then asks Qwen to produce a concise answer using the retrieved catalogue records.

## Pipeline

1. `Qwen2.5-7B-Instruct` rewrites the user request for retrieval.
2. `multilingual-e5-large` creates a query embedding.
3. FAISS returns the closest catalogue entries.
4. Metadata is loaded from SQLite and re-ranked using similarity, year and rating signals.
5. Qwen generates the final Russian-language response from the retrieved context.

The HTTP API exposes `POST /process`. [Aera Data Server](https://github.com/wxqryy/Aera_DataServer) calls this endpoint from its Redis-backed worker.

## Hardware and software

The current implementation uses 4-bit `bitsandbytes` quantization and requires:

- Python 3.11;
- an NVIDIA GPU with CUDA support;
- approximately 8 GB of VRAM;
- enough disk space for Qwen, E5, the FAISS index and the SQLite catalogue.

CPU-only execution is not supported by this version.

## Setup

Create a virtual environment. Install a CUDA-enabled PyTorch build that matches your driver, then install the remaining dependencies:

```bash
python -m venv .venv
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

If your machine uses another CUDA version, select the matching PyTorch index instead of `cu128`.

Download the runtime data:

1. [FAISS index](https://drive.google.com/file/d/1mUetoYGk3WaUapCj3jPRB16lNjtFGodq/view?usp=sharing)
2. [Movie database](https://drive.google.com/file/d/17n0-esqrE9S3z0PWok46erVp7KO820ZL/view?usp=sharing)

Place the files in the repository root:

```text
Aera_RAG/
├── faiss_index
├── database.db
├── main.py
└── requirements.txt
```

The original layout reference is also available in [`img.png`](img.png).

Start the service:

```bash
python main.py
```

It listens on `127.0.0.1:5001` by default. Paths and model identifiers can be overridden with environment variables:

| Variable | Default |
| --- | --- |
| `MODEL_ID` | `Qwen/Qwen2.5-7B-Instruct` |
| `EMBEDDING_MODEL_ID` | `intfloat/multilingual-e5-large` |
| `FAISS_INDEX_PATH` | `faiss_index` |
| `MOVIE_DATABASE_PATH` | `database.db` |
| `HOST` | `127.0.0.1` |
| `PORT` | `5001` |

## Example request

```bash
curl -X POST http://127.0.0.1:5001/process \
  -H "Content-Type: application/json" \
  -d '{"data":"Посоветуй лёгкую научную фантастику"}'
```

Model weights are downloaded from Hugging Face on the first launch. They and the catalogue files are not part of this repository.

## License

The source code is available under the [MIT License](LICENSE). The Qwen and E5 models remain subject to their own licenses.
