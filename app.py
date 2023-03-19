from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    model = pipeline("text2text-generation", model="t5-base", tokenizer="t5-base")
    input_text = request.form["input_text"]
    summary = model(input_text, max_length=150, min_length=30, do_sample=False)[0]["generated_text"]
    return render_template("result.html", summary=summary)

@app.route("/translate_hindi", methods=["POST"])
def translate_hindi():
    model = pipeline("translation_hi_to_en", model="Helsinki-NLP/opus-mt-en-hi")
    summary = request.form["summary"]
    translated_summary = model(summary, max_length=500)[0]["translation_text"]
    return render_template("result.html", summary=summary, translated_summary=translated_summary)

@app.route("/translate_telugu", methods=["POST"])
def translate_telugu():
    model = pipeline("translation_te_to_en", model="Helsinki-NLP/opus-mt-en-te")
    summary = request.form["summary"]
    translated_summary = model(summary, max_length=500)[0]["translation_text"]
    return render_template("result.html", summary=summary, translated_summary=translated_summary)

if __name__ == "__main__":
    app.run(debug=True)
