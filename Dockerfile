FROM python:3.12-slim

# LaTeX toolchain (tectonic) + ghostscript for dark-mode PDF
# + ffmpeg for video auto-split and transcode fallback.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tectonic ghostscript ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY cli.py .

ENV DATA_DIR=/app/data PORT=8000
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
