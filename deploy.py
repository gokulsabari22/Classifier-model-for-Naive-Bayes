from fastapi import FastAPI, File, UploadFile, HTTPException
import pickle
import re
from nltk.corpus import stopwords
import json
from typing import List

with open("Model", "rb") as f:
    model = pickle.load(f)

with open("Vector", "rb") as fh:
    vector = pickle.load(fh)

app = FastAPI(
    title="Document Classifier",
    description="To classify if a document is SDS or Non-SDS",
)


@app.post("/predict")
async def deploy(file: UploadFile = File(...)):
    data_list = []
    data_out = {}
    extension = file.filename[-3:]
    if extension != "txt":
        raise HTTPException(400, detail="Please upload text file")
    if extension == "txt":
        text = await file.read()
        file_size = len(text)
        if file_size >= 1000000:
            raise HTTPException(400, detail="Please uploa text file less than 1MB")
        text = text.decode("utf-8")
        stop_words = set(stopwords.words("english"))
        sentence = re.sub(r"(\\n|\\t)", " ", text)
        sentence = re.sub("[^A-Za-z]", " ", sentence)
        sentence = sentence.lower()
        sentence = sentence.split()
        sentence = [words for words in sentence if words not in stop_words]
        sentence = " ".join(sentence)
        data = [sentence]
        cv = vector.transform(data)
        label = model.predict(cv)
        classprobability = model.predict_proba(cv)
        if label == 1:
            data_out["Type"] = "SDS"
            data_out["Filename"] = file.filename
            data_out["Probability"] = classprobability.ravel()[1]
            data_out["Label"] = 1
            data_list.append(data_out)
        else:
            data_out["Type"] = "Non-SDS"
            data_out["Filename"] = file.filename
            data_out["Probability"] = classprobability.ravel()[0]
            data_out["Label"] = 0
            data_list.append(data_out)
    return data_list
