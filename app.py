from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from speech_to_text import transcribe_audio
from ollama_ai import generate_minutes, label_speakers
from moviepy import VideoFileClip

import os
import uuid

load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

ALLOWED_VIDEO_EXTENSIONS = {"mp4", "webm", "mov", "mkv", "avi"}


def allowed_video(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS
    )


def create_speaker_transcript(audio_path):
    transcript = transcribe_audio(audio_path)

    if not transcript or not transcript.strip():
        raise RuntimeError("No speech detected in the audio.")

    try:
        labeled = label_speakers(transcript.strip())
    except Exception as error:
        app.logger.exception("Speaker labeling failed")
        labeled = (
            transcript.strip()
            + "\n\n[SPEAKER ERROR: "
            + str(error)
            + "]"
        )

    return "MEETING TRANSCRIPT\n\n" + labeled


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"success": False, "error": "No audio file received."}), 400

    audio = request.files["audio"]

    if not audio.filename:
        return jsonify({"success": False, "error": "No audio file selected."}), 400

    audio_path = os.path.join(
        app.config["UPLOAD_FOLDER"], f"{uuid.uuid4()}.webm"
    )

    try:
        audio.save(audio_path)
        transcript = create_speaker_transcript(audio_path)
        return jsonify({"success": True, "transcript": transcript})

    except Exception as error:
        app.logger.exception("Audio transcription failed")
        return jsonify({"success": False, "error": str(error)}), 500

    finally:
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except OSError:
                pass


@app.route("/upload_video", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"success": False, "error": "No video file received."}), 400

    video = request.files["video"]

    if not video.filename:
        return jsonify({"success": False, "error": "No video file selected."}), 400

    if not allowed_video(video.filename):
        return jsonify({
            "success": False,
            "error": "Use MP4, WebM, MOV, MKV or AVI."
        }), 400

    extension = video.filename.rsplit(".", 1)[1].lower()

    video_path = os.path.join(
        app.config["UPLOAD_FOLDER"], f"{uuid.uuid4()}.{extension}"
    )
    audio_path = os.path.join(
        app.config["UPLOAD_FOLDER"], f"{uuid.uuid4()}.mp3"
    )

    video_clip = None

    try:
        video.save(video_path)

        video_clip = VideoFileClip(video_path)

        if video_clip.audio is None:
            return jsonify({
                "success": False,
                "error": "The video does not contain audio."
            }), 400

        video_clip.audio.write_audiofile(
            audio_path, fps=16000, bitrate="64k", logger=None
        )

        video_clip.close()
        video_clip = None

        transcript = create_speaker_transcript(audio_path)
        return jsonify({"success": True, "transcript": transcript})

    except Exception as error:
        app.logger.exception("Video processing failed")
        return jsonify({"success": False, "error": str(error)}), 500

    finally:
        if video_clip is not None:
            try:
                video_clip.close()
            except Exception:
                pass

        for path in (audio_path, video_path):
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


@app.route("/generate", methods=["POST"])
def generate():
    transcript = request.form.get("transcript", "").strip()

    if not transcript and request.is_json:
        data = request.get_json(silent=True) or {}
        transcript = data.get("transcript", "").strip()

    if not transcript:
        return jsonify({
            "success": False,
            "error": "No meeting transcript received."
        }), 400

    try:
        minutes = generate_minutes(transcript)

        if not minutes:
            raise RuntimeError("AI did not return any meeting minutes.")

        return jsonify({"success": True, "minutes": minutes})

    except Exception as error:
        app.logger.exception("Minutes generation failed")
        return jsonify({"success": False, "error": str(error)}), 500


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "success": False,
        "error": "Maximum upload size is 500 MB."
    }), 413


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )