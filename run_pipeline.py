#!/usr/bin/env python3

import argparse
import logging
import traceback
from pathlib import Path
from typing import List, Set

from pdf_to_images import render_pdf_to_images
from extract_audio import extract_and_concatenate
from transcribe_audio import transcribe_audio_file
from generate_latex import generate_latex
from latex_utils import sanitise_latex, compile_pdf, create_dark_pdf

SUPPORTED_IMAGE_EXTS: Set[str] = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
SUPPORTED_VIDEO_EXTS: Set[str] = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
SUPPORTED_AUDIO_EXTS: Set[str] = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
SUPPORTED_PDF_EXTS: Set[str] = {".pdf"}
SUPPORTED_TEXT_EXTS: Set[str] = {".txt"}

def run_pipeline(input_files: List[Path], work_path: Path, model_size: str | None = None):
    img_dir = work_path / "images"
    audio_dir = work_path / "audio"
    transcript_dir = work_path / "transcripts"
    latex_dir = work_path / "latex"

    for d in [img_dir, audio_dir, transcript_dir, latex_dir]:
        d.mkdir(parents=True, exist_ok=True)

    pdf_files = [f for f in input_files if f.suffix.lower() in SUPPORTED_PDF_EXTS]
    img_files = [f for f in input_files if f.suffix.lower() in SUPPORTED_IMAGE_EXTS]
    media_files = [f for f in input_files if f.suffix.lower() in SUPPORTED_VIDEO_EXTS or f.suffix.lower() in SUPPORTED_AUDIO_EXTS]
    text_files = [f for f in input_files if f.suffix.lower() in SUPPORTED_TEXT_EXTS]

    logging.info(f"Found {len(pdf_files)} PDF files")
    logging.info(f"Found {len(img_files)} image files")
    logging.info(f"Found {len(media_files)} media files")
    logging.info(f"Found {len(text_files)} text files")

    if pdf_files:

        for pdf in pdf_files:
            logging.info(f"  - Processing '{pdf.name}'")
            render_pdf_to_images(pdf, img_dir)
    else:
        logging.info("No PDF files found.")

    transcript = ""
    if media_files:
        logging.info(f"Found {len(media_files)} audio/video files to process...")
        
        combined_audio = audio_dir / "combined_audio.wav"
        logging.info("Extracting and combining audio tracks...")
        extract_and_concatenate(media_files, combined_audio)
        
        logging.info("Transcribing combined audio (this may take a while)...")

        transcript += transcribe_audio_file(combined_audio, model_size=model_size)

        audio_transcript_path = transcript_dir / "transcript.txt"
        audio_transcript_path.write_text(transcript, encoding="utf-8")
        logging.info(f"Transcript saved to '{audio_transcript_path}'")


    final_image_paths = sorted(list(img_dir.glob("*.*"))) + img_files
    final_images: List[bytes] = []
    for image_path in final_image_paths:
        try:
            final_images.append(image_path.read_bytes())
        except Exception:
            traceback.print_exc()

    final_text = transcript

    for text_file in text_files:
        try:
            final_text += text_file.read_text(encoding="utf-8", errors="replace") + "\n\n"
        except Exception:
            traceback.print_exc()
            
    final_text = transcript + '\n' + final_text.strip()
    
    if not final_images and not final_text:
        logging.warning("No content available to generate the document. Exiting.")
        return

    logging.info("Generating LaTeX document via generative model...")
    raw_latex = generate_latex(
        text=final_text,
        images=final_images
    )
    
    if not raw_latex.strip():
        logging.error("The generative model produced no LaTeX output. Exiting.")
        return

    logging.info("Sanitising and saving LaTeX file...")
    sanitised_content = sanitise_latex(raw_latex)
    tex_path = latex_dir / "lecture_notes.tex"
    tex_path.write_text(sanitised_content, encoding="utf-8")

    logging.info("Compiling LaTeX file to PDF...")
    try:
        pdf_path = compile_pdf(tex_path)
        logging.info(f"PDF successfully generated: {pdf_path.resolve()}")
        create_dark_pdf(pdf_path)
    except Exception as e:
        logging.error(f"Error during PDF compilation: {e}")
        logging.error("The .tex file is still available here: %s", tex_path.resolve())


def main():
    parser = argparse.ArgumentParser(
        description="A complete pipeline to generate lecture notes from input files (PDFs, videos, audio, txt)."
    )
    parser.add_argument(
        "input_files",
        type=Path,
        nargs='+',
        help="One or more paths to input files (PDFs, videos, audio, txt)."
    )
    parser.add_argument(
        "work_folder",
        type=Path,
        help="Path to the working folder for artifacts and the final output."
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default=None,
        choices=["tiny", "base", "small", "medium", "large", "large-v2"],
        help="Whisper model size to use for transcription. Defaults to 'medium' if not specified."
    )
    
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    resolved_input_files = [p.resolve() for p in args.input_files]
    input_files_exist = [f for f in resolved_input_files if f.is_file()]
    
    if not input_files_exist:
        logging.error("No valid input files provided.")
        return

    run_pipeline(input_files_exist, args.work_folder.resolve(), model_size=args.model_size)

if __name__ == "__main__":
    main()
