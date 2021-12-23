import pickle
import re
from nltk.corpus import stopwords
import json
 


with open("Model", "rb") as f:
    model = pickle.load(f)

with open("Vector", "rb") as fh:
    vector = pickle.load(fh)



def lambda_handler(event, context):
    try:
        text = event["text"]
        stop_words = set(stopwords.words("english"))
        sentence = re.sub(r"(\\n|\\t)", " ", text)
        sentence = re.sub("[^A-Za-z]", " ", sentence)
        sentence = sentence.lower()
        sentence = sentence.split()
        sentence = [words for words in sentence if words not in stop_words]
        sentence = ' '.join(sentence)
        data = [sentence]
        cv = vector.transform(data)
        label = model.predict(cv)
        classprobability = model.predict_proba(cv)
        if label == 1:
            return {"File": "SDS", "Probability": classprobability.ravel()[1], "Label": 1}
        else:
            pass
        return {"File": "Non-SDS", "Probability": classprobability.ravel()[0], "Label": 0}
    except (Exception, KeyboardInterrupt) as e:
        return "Error occurred"
