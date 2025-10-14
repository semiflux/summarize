import subprocess
from pathlib import Path
import tempfile

def extract_audio(input_path: Path, output_path: Path):
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        "-vn", "-loglevel", "error", str(output_path)
    ]
    subprocess.run(cmd, check=True)

def concatenate_audios(audio_paths: list[Path], output_path: Path):
    tmp_list = output_path.parent / "concat_list.txt"
    with tmp_list.open("w", encoding="utf-8") as f:
        for p in audio_paths:
            f.write(f"file '{p.resolve()}'\n")
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(tmp_list),
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        "-loglevel", "error",
        str(output_path)
    ], check=True)
    tmp_list.unlink()

def extract_and_concatenate(input_paths: list[Path], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        temp_files = []

        for i, path in enumerate(input_paths, start=1):
            temp_wav = tmpdir_path / f"temp_{i:03d}.wav"
            extract_audio(path, temp_wav)
            temp_files.append(temp_wav)

        concatenate_audios(temp_files, output_path)

    return output_path
