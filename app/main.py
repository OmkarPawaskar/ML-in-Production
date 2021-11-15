import pathlib
import json
from typing import Optional
from fastapi import FastAPI
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

app = FastAPI()
BASE_DIR = pathlib.Path(__file__).resolve().parent #fastapi base dir

MODEL_DIR = BASE_DIR.parent / "models" 
SMS_SPAM_MODEL_PATH = MODEL_DIR / "spam-sms"
MODEL_PATH = SMS_SPAM_MODEL_PATH / "spam-model.h5"
TOKENIZER_PATH = SMS_SPAM_MODEL_PATH / "spam-classifer-tokenizer.json"
METADATA_PATH = SMS_SPAM_MODEL_PATH / "spam-classifer-metadata.json"
AI_MODEL = None
AI_TOKENIZER = None
MODEL_METADATA = {}
LABELS_LEGEND_INVERTED = {}

@app.on_event("startup")
def on_startup():
    global AI_MODEL, AI_TOKENIZER, MODEL_METADATA, LABELS_LEGEND_INVERTED
    #load my model
    if MODEL_PATH.exists():
        AI_MODEL = load_model(MODEL_PATH)
    if TOKENIZER_PATH.exists():
        t_json = TOKENIZER_PATH.read_text()
        AI_TOKENIZER = tokenizer_from_json(t_json)
    if METADATA_PATH.exists():
        MODEL_METADATA = json.loads(METADATA_PATH.read_text())
        LABELS_LEGEND_INVERTED = MODEL_METADATA["labels_legend_inverted"]

def predict(query: str):
    sequences = AI_TOKENIZER.texts_to_sequences([query])
    max_len = MODEL_METADATA.get('max_sequence') or 300
    x_input = pad_sequences(sequences, maxlen=max_len)
    preds_array = AI_MODEL.predict(x_input)
    preds = preds_array[0]
    top_index_val = np.argmax(preds)
    top_pred = {"label" : LABELS_LEGEND_INVERTED[str(top_index_val)], "confidence" : float(preds[top_index_val])}
    labeled_preds = [{"label" : LABELS_LEGEND_INVERTED[str(i)], "confidence" : float(x)} for i, x in enumerate(list(preds))]
    return {"top" : top_pred, "predictions" : labeled_preds}
    

@app.get('/') # /?q = Hello world
def index(q : Optional[str] = None):
    global AI_MODEL, MODEL_METADATA
    query = q or "hello world"
    preds_dict = predict(query)
    return {"query" : query, "results" : preds_dict}