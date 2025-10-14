import subprocess
import math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from faster_whisper import WhisperModel
import json
from typing import List
import tempfile
import shutil

def get_audio_duration(audio_path: Path) -> float:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(audio_path)],
            check=True, capture_output=True, text=True
        )
        streams = json.loads(result.stdout).get("streams", [{}])
        return float(streams[0].get("duration", 0.0))
    except Exception:
        return 0.0

def transcribe_chunk(chunk_path: Path, model_size: str, device: str, compute_type: str) -> str:
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, _ = model.transcribe(str(chunk_path), beam_size=5)
    return "".join(segment.text for segment in segments).strip()

def transcribe_audio_parallel(
    input_path: Path,
    num_workers: int = 3,
    model_size: str = "medium",
    device: str = "cpu",
    compute_type: str = "int8",
    chunk_seconds: int = 600
) -> str:
    duration = get_audio_duration(input_path)
    if duration <= 0:
        raise ValueError(f"Invalid audio duration: {input_path}")
    num_chunks = math.ceil(duration / chunk_seconds)
    all_texts: List[str] = []

    with tempfile.TemporaryDirectory() as tmpdir:
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

        worker_func = partial(transcribe_chunk, model_size=model_size, device=device, compute_type=compute_type)
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            for text in ex.map(worker_func, chunk_paths):
                all_texts.append(text)

    return " ".join(all_texts)

def transcribe_audio_file(
    input_path: Path,
    num_workers: int = 3,
    model_size: str = "medium",
    device: str = "cpu",
    compute_type: str = "int8"
) -> str:
    return transcribe_audio_parallel(input_path, num_workers, model_size, device, compute_type)
