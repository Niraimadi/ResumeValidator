from flask import Flask, request
from llama_index.llms.ollama import Ollama
from werkzeug.utils import secure_filename
from document_ingestion import ingest_data
import os
import json

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/validate_resume", methods=['POST'])
def validate_resume():
    job_desc = request.form.get("Job Description")
    file = request.files["Resume"]

    if not os.path.exists("temp"):
      os.makedirs("temp")
      print("Directory created successfully!")
    else:
        print("Directory already exists!")

    filename = secure_filename(file.filename)
    filepath = os.path.join("temp", filename)
    file.save(filepath)

    response = ingest_data(filepath, job_desc)
    print(response)
    return response

if __name__ == "__main__":
    app.run(debug=True)