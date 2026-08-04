# 🎥 RAG-Based Video Q&A Assistant

A local, privacy-friendly **Retrieval-Augmented Generation (RAG)** system that lets you ask natural-language questions about the content of your own videos — and get back an answer citing the **exact source video and timestamp**.

No cloud APIs, no data leaving your machine. Everything (speech-to-text, embeddings, and answer generation) runs locally via [Ollama](https://ollama.com) and [Whisper](https://github.com/openai/whisper).

---

## ✨ Features

- 🎞️ Converts any video library into searchable, timestamped transcripts
- 🗣️ Hindi (or any spoken language) → English translation built into transcription
- 🧩 Groups raw transcript segments into richer, context-aware chunks
- 🔎 Semantic search over your entire video library using FAISS
- 🤖 Answers grounded strictly in retrieved transcript context (no hallucination)
- 📍 Every answer cites the **source video name + MM:SS timestamp**
- ♻️ Fully incremental — re-running the pipeline only processes newly added videos

---

## 🏗️ Architecture

```
videos/*.mp4|.mkv|.mov|.avi
      │
      │  process_video_1.py        →  FFmpeg extracts audio
      ▼
audios/*.mp3
      │
      │  create_chunk_of_audios_2.py →  Whisper transcribes + translates
      │                                  + merges small segments into
      │                                  richer 5-segment chunks
      ▼
jsons/*.json   (source_name, start, end, text)
      │
      │  create_embeddings_3.py     →  Ollama (nomic-embed-text) embeds
      │                                  each chunk
      ▼
embeddings.joblib   (chunk metadata + vectors)
video_index.faiss   (FAISS IndexFlatIP — cosine similarity search)
      │
      │  process_question_4.py      →  question → embed → FAISS search (k=5)
      ▼
Top-5 relevant chunks → context → Ollama (gemma2:2b) → grounded answer
```

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Video → Audio | FFmpeg |
| Speech-to-Text + Translation | OpenAI Whisper (`small`) |
| Embedding Model | `nomic-embed-text` (via Ollama) |
| Vector Search | FAISS (`IndexFlatIP`) |
| Answer Generation | `gemma2:2b` (via Ollama) |
| Metadata Storage | pandas + joblib |

---

## 📁 Project Structure

```
.
├── process_video_1.py            # Step 1: extract audio from videos
├── create_chunk_of_audios_2.py   # Step 2: transcribe, translate, merge chunks
├── create_embeddings_3.py        # Step 3: generate embeddings + FAISS index
├── process_question_4.py         # Step 4: ask questions, get grounded answers
│
├── videos/                       # 📥 put your source videos here
├── audios/                       # generated .mp3 files (auto-created)
├── jsons/                        # generated transcript chunks (auto-created)
├── embeddings.joblib             # chunk metadata + embeddings (auto-created)
├── video_index.faiss             # FAISS vector index (auto-created)
├── prompt.txt                    # last prompt sent to the LLM (for debugging)
└── response.txt                  # last generated answer
```

---

## ⚙️ Setup

### 1. Prerequisites

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/download.html) installed and accessible
- [Ollama](https://ollama.com) installed and running

### 2. Install Python dependencies

```bash
pip install openai-whisper faiss-cpu pandas numpy joblib requests
```

### 3. Pull the required Ollama models

```bash
ollama pull nomic-embed-text
ollama pull gemma2:2b
```

### 4. Configure FFmpeg path

Open `process_video_1.py` and set `ffmpeg_path` to your local FFmpeg binary path.

---

## 🚀 Usage

Drop your videos into the `videos/` folder, then run the pipeline in order:

```bash
# 1. Extract audio from videos
python process_video_1.py

# 2. Transcribe, translate, and chunk the audio
python create_chunk_of_audios_2.py

# 3. Generate embeddings and build the FAISS index
python create_embeddings_3.py

# 4. Ask questions!
python process_question_4.py
```

Each script only processes **new** files — already-processed videos are automatically skipped, so you can safely re-run the pipeline any time you add new videos.

---

## 🧠 How It Works

1. **Transcription** — Whisper transcribes each video's audio and translates it to English, producing short, timestamped segments.
2. **Chunk Merging** — Consecutive raw segments (1–2 sec each) are merged into groups of 5 to give the embedding model richer, more meaningful context per chunk.
3. **Embedding** — Each merged chunk is embedded using `nomic-embed-text` and stored in a FAISS `IndexFlatIP` index (cosine similarity via L2-normalized vectors).
4. **Retrieval** — A user's question is embedded the same way and matched against the index to retrieve the top-5 most relevant chunks.
5. **Generation** — The retrieved chunks are passed as context to `gemma2:2b`, which is instructed to answer *only* from that context and always cite the source video and timestamp.

---

## ⚠️ Known Limitations

- Answer generation uses a small 2B-parameter local model — retrieval is strong, but generation fluency can occasionally be limited.
- Fixed top-k (5) retrieval regardless of query complexity.
- Single-user, local-only — no auth, no multi-user support.
- No formal retrieval/answer evaluation metrics yet.

## 🔮 Future Improvements

- Hybrid search (keyword + semantic)
- Evaluation harness for retrieval recall / answer faithfulness
- Streamlit front-end for asking questions and jumping to timestamps
- Swap in a larger/hosted model for higher-quality generation

---

## 📄 License

MIT — feel free to use and modify.

