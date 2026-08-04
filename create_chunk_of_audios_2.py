import os
import json
import whisper

GROUP_SIZE = 5  # kitne raw whisper chunks ek bade chunk me merge karne hain

model = whisper.load_model("small")
os.makedirs("jsons", exist_ok=True)


def merge_chunks(chunks, group_size=GROUP_SIZE):
    
    merged = []

    for i in range(0, len(chunks), group_size):
        group = chunks[i:i + group_size]
        if not group:
            continue

        merged_text = " ".join(c["text"].strip() for c in group).strip()

        merged.append({
            "source_name": group[0]["source_name"],
            "source_type": group[0]["source_type"],
            "start": group[0]["start"],
            "end": group[-1]["end"],
            "text": merged_text
        })

    return merged


for audio in os.listdir("audios"):
    if not audio.endswith(".mp3"):
        continue

    title = os.path.splitext(audio)[0]
    safe_name = title.replace(" ", "_").replace("&", "and")
    json_path = f"jsons/{safe_name}.json"

    if os.path.exists(json_path):
        print(f"Skipping {audio}")
        continue

    print(f"Transcribing: {audio}")

    try:
        result = model.transcribe(
            audio=f"audios/{audio}",
            language="hi",
            task="translate",
            fp16=False
        )

    except Exception as e:
        print(f"Error processing {audio}")
        print(e)
        continue

    # Step 1: Whisper ke raw, chhote (1-2 sec) segments se chunks banao
    raw_chunks = []
    for segment in result["segments"]:
        raw_chunks.append({
            "source_name": title,
            "source_type": "video",
            "start": round(segment['start'], 2),
            "end": round(segment['end'], 2),
            "text": segment["text"].strip()
        })

    # Step 2: In raw chunks ko groups of GROUP_SIZE me merge karo
    merged_chunks = merge_chunks(raw_chunks, GROUP_SIZE)

    print(f"  {len(raw_chunks)} raw chunks -> {len(merged_chunks)} merged chunks")

    full_text_with_chunks = {
        "text": result["text"],
        "chunks": merged_chunks
    }

    with open(f"jsons/{safe_name}.json", "w", encoding="utf-8") as f:
        json.dump(full_text_with_chunks, f, ensure_ascii=False, indent=4)

print("All audios converted to merged JSON chunks!")