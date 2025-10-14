#!/usr/bin/env python3
import argparse
import logging
import shutil
from pathlib import Path

from pdf_to_images import render_pdf_to_images
from extract_audio import extract_and_concatenate
from transcribe_audio import transcribe_audio_file
from generate_latex import generate_latex
from latex_utils import sanitise_latex, compile_pdf, create_dark_pdf

SUPPORTED_IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".gif"]
SUPPORTED_VIDEO_EXTS = [".mp4", ".mov", ".avi", ".mkv", ".webm"]
SUPPORTED_AUDIO_EXTS = [".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"]
SUPPORTED_PDF_EXTS = [".pdf"]
SUPPORTED_TEXT_EXTS = [".txt"]

def run_pipeline(input_path: Path, work_path: Path):
    img_dir = work_path / "images"
    audio_dir = work_path / "audio"
    transcript_dir = work_path / "transcripts"
    latex_dir = work_path / "latex"

    for d in [img_dir, audio_dir, transcript_dir, latex_dir]:
        d.mkdir(parents=True, exist_ok=True)

    all_files = list(input_path.rglob("*"))
    pdf_files = [f for f in all_files if f.suffix.lower() in SUPPORTED_PDF_EXTS]
    img_files = [f for f in all_files if f.suffix.lower() in SUPPORTED_IMAGE_EXTS]
    video_files = [f for f in all_files if f.suffix.lower() in SUPPORTED_VIDEO_EXTS]
    audio_files = [f for f in all_files if f.suffix.lower() in SUPPORTED_AUDIO_EXTS]
    text_files = [f for f in all_files if f.suffix.lower() in SUPPORTED_TEXT_EXTS]

    if pdf_files:
        logging.info(f"Found {len(pdf_files)} PDF files to convert to images...")
        for pdf in pdf_files:
            logging.info(f"  - Processing '{pdf.name}'")
            render_pdf_to_images(pdf, img_dir)
    else:
        logging.info("No PDF files found.")

    if img_files:
        logging.info(f"Found {len(img_files)} image files to copy...")
        for img in img_files:
            shutil.copy(img, img_dir / img.name)
    else:
        logging.info("No image files found.")

    media_files = video_files + audio_files
    if media_files:
        logging.info(f"Found {len(media_files)} audio/video files to process...")
        
        combined_audio = audio_dir / "combined_audio.wav"
        logging.info("Extracting and combining audio tracks...")
        extract_and_concatenate(media_files, combined_audio)
        
        logging.info("Transcribing combined audio (this may take a while)...")
        transcript_text = transcribe_audio_file(combined_audio)
        transcript_path = transcript_dir / "audio_transcript.txt"
        transcript_path.write_text(transcript_text, encoding="utf-8")
        logging.info(f"Transcript saved to '{transcript_path}'")
    else:
        logging.info("No audio/video files found.")

    if text_files:
        logging.info(f"Found {len(text_files)} text files to copy...")
        for txt in text_files:
            shutil.copy(txt, transcript_dir / txt.name)
    else:
        logging.info("No text files found.")

    final_images = sorted(list(img_dir.glob("*.*")))
    final_transcripts = sorted(list(transcript_dir.glob("*.txt")))

    if not final_images and not final_transcripts:
        logging.warning("No content (images or text) available to generate the document. Exiting.")
        return

    logging.info("Generating LaTeX document via generative model...")
    raw_latex = generate_latex(
        transcript_files=final_transcripts,
        image_files=final_images
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    parser = argparse.ArgumentParser(
        description="A complete pipeline to generate lecture notes from PDF, audio, and video files."
    )
    parser.add_argument(
        "input_folder",
        type=Path,
        help="Path to the folder containing input files (PDFs, videos, audio, txt)."
    )
    parser.add_argument(
        "work_folder",
        type=Path,
        help="Path to the working folder for artifacts and the final output."
    )
    args = parser.parse_args()

    if not args.input_folder.is_dir():
        logging.error(f"Input folder does not exist: {args.input_folder}")
        return

    run_pipeline(args.input_folder.resolve(), args.work_folder.resolve())

if __name__ == "__main__":
    main()
