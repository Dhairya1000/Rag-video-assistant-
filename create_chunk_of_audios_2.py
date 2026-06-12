import os
import json
import whisper

model = whisper.load_model("small")
os.makedirs("jsons", exist_ok=True)

for audio in os.listdir("audios"):
    if not audio.endswith(".mp3"):
        continue

    title = os.path.splitext(audio)[0]

    print(f"Transcribing: {audio}")

    result = model.transcribe(
        audio=f"audios/{audio}",
        language="hi",
        task="translate",
        fp16=False
    )

    chunks = []
    for segment in result["segments"]:
        chunks.append({
            "source_name": title,
            "source_type": "video",
            "start": round(segment['start'], 2),
            "end": round(segment['end'], 2),
            "text": segment["text"].strip()
})

    full_text_with_chunks = {
        "text": result["text"],
        "chunks": chunks
    }

    safe_name = audio.replace(" ", "_").replace("&", "and")

    with open(f"jsons/{safe_name}.json", "w", encoding="utf-8") as f:
        json.dump(full_text_with_chunks, f, ensure_ascii=False, indent=4)

print("✅ All audios converted to JSON chunks!")
