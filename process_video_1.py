import os
import subprocess

video_dir = "videos"
audio_dir = "audios"

os.makedirs(audio_dir, exist_ok=True)

ffmpeg_path = r"D:\AI_projects\Rag_project\video_assistant\ffmpeg-master-latest-win64-gpl-shared\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"


for file in os.listdir(video_dir):
    if not file.lower().endswith(('.mp4', '.mkv', '.mov', '.avi')):
        continue

    source_name = os.path.splitext(file)[0]

    output_name = f"{source_name}.mp3"
    output_path = os.path.join(audio_dir, output_name)

    if os.path.exists(output_path):
        print(f"Already converted: {output_name}")
        continue

    print(f"Converting: {file} → {output_name}")

    subprocess.run([
        ffmpeg_path,
        "-i", os.path.join(video_dir, file),
        "-vn",
        "-acodec", "mp3",
        output_path
    ])

print("✅ All videos converted!")
