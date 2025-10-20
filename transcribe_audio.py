from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import math
import tempfile
from faster_whisper import WhisperModel
from typing import List
import json

def get_audio_duration(audio_path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(audio_path)],
        check=True, capture_output=True, text=True
    )
    streams = json.loads(result.stdout).get("streams", [{}])
    return float(streams[0].get("duration", 0.0))

def transcribe_chunk(model: WhisperModel, chunk_path: Path) -> str:
    segments, _ = model.transcribe(str(chunk_path), beam_size=5)
    return "".join(segment.text for segment in segments).strip()

def transcribe_audio_file(
    input_path: Path,
    model_size: str = "medium",
    device: str = "cpu",
    compute_type: str = "int8",
    num_workers_model: int = 4,
    num_workers_python: int = 3,
    chunk_seconds: int = 600
) -> str:
    duration = get_audio_duration(input_path)
    if duration <= 0:
        raise ValueError(f"Invalid audio duration: {input_path}")
    num_chunks = math.ceil(duration / chunk_seconds)
    
    model = WhisperModel(model_size, device=device, compute_type=compute_type, num_workers=num_workers_model)
    
    all_texts: List[str] = []

    temp_dir_obj = tempfile.TemporaryDirectory()
    tmpdir = temp_dir_obj.name
    temp_dir_obj.cleanup = lambda: None

    chunks_dir = Path(tmpdir)
    chunk_paths: List[Path] = []
    
    for i in range(num_chunks):
        start = i * chunk_seconds
        chunk_path = chunks_dir / f"chunk_{i:03d}.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(input_path),
            "-ss", str(start), "-t", str(chunk_seconds),
            "-c:a", "pcm_s16le", "-loglevel", "error", str(chunk_path)
        ], check=True)
        chunk_paths.append(chunk_path)
    
    with ThreadPoolExecutor(max_workers=num_workers_python) as ex:
        for text in ex.map(lambda p: transcribe_chunk(model, p), chunk_paths):
            all_texts.append(text)

    return " ".join(all_texts)
