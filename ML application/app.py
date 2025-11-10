from flask import Flask, request, render_template
from transformers import pipeline

app = Flask(__name__)

# Load Hugging Face sentiment model (tiny)
sentiment = pipeline("sentiment-analysis")

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    score = None
    if request.method == "POST":
        text = request.form["text"]
        output = sentiment(text)[0]
        result = output["label"]
        score = round(output["score"] * 100, 2)
    return render_template("index.html", result=result, score=score)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
