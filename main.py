import os



_FFMPEG_BIN = os.environ.get("FFMPEG_BIN")
if _FFMPEG_BIN:
    os.environ["PATH"] = _FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")

from flask import Flask, request, send_from_directory, jsonify
import uuid
import re
import time
import logging
from logging.handlers import RotatingFileHandler
import traceback
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
_console = logging.StreamHandler()
_console.setFormatter(logging.Formatter(_LOG_FORMAT))
_file = RotatingFileHandler("server.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
_file.setFormatter(logging.Formatter(_LOG_FORMAT))
logging.basicConfig(level=logging.INFO, handlers=[_console, _file])
logging.getLogger("werkzeug").setLevel(logging.INFO)


@app.before_request
def _log_request_start():
    request._start_time = time.time()
    logging.info(
        f"--> {request.method} {request.path} from {request.remote_addr} "
        f"args={dict(request.args)} files={list(request.files.keys())} "
        f"content_length={request.content_length}"
    )


@app.after_request
def _log_request_end(response):
    elapsed = time.time() - getattr(request, "_start_time", time.time())
    logging.info(f"<-- {request.method} {request.path} {response.status_code} ({elapsed:.2f}s)")
    return response


@app.errorhandler(Exception)
def _handle_uncaught(err):
    logging.error(f"Unhandled error on {request.method} {request.path}: {err}")
    logging.error(traceback.format_exc())
    return jsonify({"error": "internal_server_error", "detail": str(err)}), 500

_WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL", "base")
_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE", "int8")  # int8 = CPU-friendly
_BEAM_SIZE = int(os.environ.get("WHISPER_BEAM_SIZE", "1"))
_PARALLEL_WORKERS = int(os.environ.get("PARALLEL_WORKERS", "1"))
_PARALLEL_CHUNK_SECONDS = int(os.environ.get("PARALLEL_CHUNK_SECONDS", "30"))
_MAX_SUBTITLE_SECONDS = float(os.environ.get("MAX_SUBTITLE_SECONDS", "2"))
# Divide CPU cores across workers so concurrent transcribes don't fight for threads.
_CPU_COUNT = os.cpu_count() or 2
_CPU_THREADS = max(1, _CPU_COUNT // _PARALLEL_WORKERS)
logging.info(f"Loading faster-whisper model ({_WHISPER_MODEL_NAME}, {_COMPUTE_TYPE}, "
             f"cpu_threads={_CPU_THREADS}, num_workers={_PARALLEL_WORKERS})...")
WHISPER_MODEL = WhisperModel(
    _WHISPER_MODEL_NAME,
    device="cpu",
    compute_type=_COMPUTE_TYPE,
    cpu_threads=_CPU_THREADS,
    num_workers=_PARALLEL_WORKERS,
)
logging.info("Whisper model loaded.")

def safe_filename(filename):
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', filename)

# Whisper uses a single 'zh' code for Chinese; Google Translate needs zh-CN / zh-TW.
_WHISPER_LANG_MAP = {
    'zh-cn': 'zh', 'zh-tw': 'zh', 'zh-hans': 'zh', 'zh-hant': 'zh',
}
_TRANSLATE_LANG_MAP = {
    'zh': 'zh-CN', 'zh-hans': 'zh-CN', 'zh-hant': 'zh-TW',
    'zh-cn': 'zh-CN', 'zh-tw': 'zh-TW',
}

def to_whisper_lang(code):
    return _WHISPER_LANG_MAP.get(code, code)

def to_translate_lang(code):
    return _TRANSLATE_LANG_MAP.get(code, code)

def extract_audio_from_video(video_path, audio_output):
    cmd = ['ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', audio_output, '-y']
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed: {result.stderr}")
    logging.info("Audio extracted.")

def _ts(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def generate_srt(segments, srt_output):
    with open(srt_output, "w", encoding="utf-8") as f:
        for idx, seg in enumerate(segments, 1):
            f.write(f"{idx}\n{_ts(seg['start'])} --> {_ts(seg['end'])}\n{seg['text'].strip()}\n\n")

def _probe_duration(audio_path):
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
           '-of', 'default=noprint_wrappers=1:nokey=1', audio_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return float(result.stdout.strip())


def _split_audio_chunks(audio_path, chunk_seconds, out_dir, base):
    """Slice audio into chunk_seconds files. Returns list of (offset_seconds, chunk_path)."""
    duration = _probe_duration(audio_path)
    chunks = []
    idx = 0
    offset = 0.0
    while offset < duration:
        length = min(chunk_seconds, duration - offset)
        chunk_path = os.path.join(out_dir, f"chunk_{idx:04d}_{base}.wav")
        cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
               '-ss', f"{offset}", '-t', f"{length}",
               '-i', audio_path, chunk_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg chunk split failed at offset {offset}: {result.stderr}")
        chunks.append((offset, chunk_path))
        offset += chunk_seconds
        idx += 1
    return chunks


def _transcribe_chunk(chunk_path, offset, language):
    segments_iter, _info = WHISPER_MODEL.transcribe(
        chunk_path,
        language=language,
        beam_size=_BEAM_SIZE,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        word_timestamps=True,
        condition_on_previous_text=False,
    )
    out = []
    for s in segments_iter:
        out.append({
            'start': s.start + offset,
            'end': s.end + offset,
            'text': s.text,
            'words': [{'start': w.start + offset, 'end': w.end + offset, 'word': w.word}
                      for w in (s.words or [])],
        })
    return out


def _split_long_segments(segments, max_duration):
    """Break segments longer than max_duration into shorter cues at word boundaries."""
    out = []
    for seg in segments:
        words = seg.get('words') or []
        if not words or (seg['end'] - seg['start']) <= max_duration:
            out.append({'start': seg['start'], 'end': seg['end'], 'text': seg['text'].strip()})
            continue
        bucket = []
        bucket_start = words[0]['start']
        for w in words:
            if bucket and (w['end'] - bucket_start) > max_duration:
                text = ''.join(x['word'] for x in bucket).strip()
                out.append({'start': bucket_start, 'end': bucket[-1]['end'], 'text': text})
                bucket = []
                bucket_start = w['start']
            bucket.append(w)
        if bucket:
            text = ''.join(x['word'] for x in bucket).strip()
            out.append({'start': bucket_start, 'end': bucket[-1]['end'], 'text': text})
    return out


def transcribe_parallel(audio_path, language, chunk_seconds, workers, tmp_dir, base):
    """Split audio, transcribe chunks concurrently, return merged segments in video-time order."""
    if workers <= 1:
        # Fast path: skip chunking overhead; one transcribe uses all CPU cores.
        logging.info(f"Transcribing whole audio (workers={workers}, no chunking).")
        return _transcribe_chunk(audio_path, 0.0, language)
    chunks = _split_audio_chunks(audio_path, chunk_seconds, tmp_dir, base)
    logging.info(f"Split audio into {len(chunks)} chunks of ~{chunk_seconds}s; transcribing with {workers} workers.")
    results = [None] * len(chunks)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            fut_to_idx = {
                pool.submit(_transcribe_chunk, path, offset, language): i
                for i, (offset, path) in enumerate(chunks)
            }
            for fut in as_completed(fut_to_idx):
                i = fut_to_idx[fut]
                results[i] = fut.result()
                logging.info(f"Chunk {i + 1}/{len(chunks)} transcribed.")
    finally:
        for _, path in chunks:
            try:
                os.remove(path)
            except OSError:
                pass
    return [seg for chunk_segs in results for seg in chunk_segs]

def translate_segments(segments, target_lang, source_lang='auto'):
    texts = [s['text'].strip() for s in segments]
    try:
        translator = GoogleTranslator(source=source_lang, target=target_lang)
        translated = translator.translate_batch(texts)
        translated = [t if t else original for t, original in zip(translated, texts)]
    except Exception as e:
        logging.error(f"Translation to {target_lang} failed, falling back to source text: {e}")
        translated = texts
    return [{**seg, 'text': new_text} for seg, new_text in zip(segments, translated)]

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/process_video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        logging.warning("Rejected: no 'video' field in multipart body.")
        return jsonify({'error': "missing_field", 'detail': "expected multipart field 'video'"}), 400

    video = request.files['video']
    if video.filename == '':
        logging.warning("Rejected: empty filename.")
        return jsonify({'error': 'empty_filename'}), 400

    # Read language params from the same multipart form-data body as the video file.
    video_language = (request.form.get('video_language') or '').strip().lower()
    translate_language = (request.form.get('translate_language') or '').strip().lower()

    if not video_language:
        return jsonify({'error': 'missing_param', 'detail': "form field 'video_language' is required (e.g. 'en', 'es')"}), 400
    if not translate_language:
        return jsonify({'error': 'missing_param', 'detail': "form field 'translate_language' is required (e.g. 'en', 'es')"}), 400

    base = str(uuid.uuid4()) + '-' + safe_filename(video.filename)
    video_path = os.path.join(PROCESSED_FOLDER, base)
    audio_path = os.path.join(UPLOAD_FOLDER, "audio_" + base + ".wav")
    srt_path = os.path.join(UPLOAD_FOLDER, f"{translate_language}_{base}.srt")

    try:
        logging.info(f"Saving upload -> {video_path} (video_language={video_language}, translate_language={translate_language})")
        video.save(video_path)

        extract_audio_from_video(video_path, audio_path)

        whisper_lang = to_whisper_lang(video_language)
        logging.info(f"Transcribing in parallel (chunk={_PARALLEL_CHUNK_SECONDS}s, workers={_PARALLEL_WORKERS}, language={whisper_lang})...")
        segments = transcribe_parallel(
            audio_path,
            language=whisper_lang,
            chunk_seconds=_PARALLEL_CHUNK_SECONDS,
            workers=_PARALLEL_WORKERS,
            tmp_dir=UPLOAD_FOLDER,
            base=base,
        )
        logging.info(f"Transcribed {len(segments)} segments across parallel chunks.")

        segments = _split_long_segments(segments, _MAX_SUBTITLE_SECONDS)
        logging.info(f"Split into {len(segments)} short cues (max {_MAX_SUBTITLE_SECONDS}s each).")

        src_translate = to_translate_lang(video_language)
        tgt_translate = to_translate_lang(translate_language)
        if tgt_translate == src_translate:
            logging.info("Source and target languages match, skipping translation.")
            final_segments = segments
        else:
            logging.info(f"Translating {src_translate} -> {tgt_translate}...")
            final_segments = translate_segments(segments, target_lang=tgt_translate, source_lang=src_translate)

        generate_srt(final_segments, srt_path)

        return jsonify({
            'message': 'Video processed successfully',
            'video_language': video_language,
            'translate_language': translate_language,
            'video_url': f'/download/{os.path.basename(video_path)}',
            'subtitle_url': f'/subtitles/{os.path.basename(srt_path)}',
        })
    except Exception as e:
        logging.error(f"process_video failed for {base}: {e}")
        logging.error(traceback.format_exc())
        return jsonify({'error': 'processing_failed', 'detail': str(e)}), 500

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

@app.route('/subtitles/<filename>', methods=['GET'])
def download_subtitle(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, mimetype='application/x-subrip')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
