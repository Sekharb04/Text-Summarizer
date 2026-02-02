import os
import warnings
from flask import Flask, render_template, request, send_file
from preprocess import preprocess_text
from summarizer import generate_summary
from textrank import textrank_summary
from confidence import confidence_score
from confidence import confidence_label
import io

# Suppress TensorFlow and other warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    original_text = ""
    abstractive_summary = ""
    extractive_summary = ""
    abstractive_confidence = 0
    extractive_confidence = 0
    abstractive_label = ""
    extractive_label = ""
    summary_type = "both"
    summary_length = 3  # default number of sentences/words

    if request.method == "POST":
        original_text = request.form["text"]
        summary_type = request.form.get("summary_type", "both")
        summary_length = int(request.form.get("summary_length", 3))

        cleaned_text = preprocess_text(original_text)

        if summary_type in ["abstractive", "both"]:
            abstractive_summary = generate_summary(
                cleaned_text, summary_length)
            abstractive_confidence = confidence_score(
                original_text, abstractive_summary)
            abstractive_label = confidence_label(abstractive_confidence)

        if summary_type in ["extractive", "both"]:
            extractive_summary = textrank_summary(
                original_text, summary_length)
            extractive_confidence = confidence_score(
                original_text, extractive_summary)
            extractive_label = confidence_label(extractive_confidence)

    return render_template(
        "index.html",
        original_text=original_text,
        abstractive_summary=abstractive_summary,
        extractive_summary=extractive_summary,
        abstractive_confidence=abstractive_confidence,
        extractive_confidence=extractive_confidence,
        abstractive_label=abstractive_label,
        extractive_label=extractive_label,
        summary_type=summary_type,
        summary_length=summary_length
    )


@app.route("/download/<summary_type>")
def download_summary(summary_type):
    summary_text = request.args.get("text", "")
    filename = f"{summary_type}_summary.txt"
    return send_file(
        io.BytesIO(summary_text.encode()),
        mimetype="text/plain",
        as_attachment=True,
        download_name=filename
    )


if __name__ == "__main__":
    print("🚀 Starting Advanced Text Summarizer...")
    print("📚 Loading AI models and preparing summarization engines...")
    print("✅ Server ready! Open http://127.0.0.1:5000 in your browser")
    print("=" * 60)
    app.run(debug=True)
