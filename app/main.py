import pathlib
import json
from typing import Optional
from fastapi import FastAPI

from . import ml


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
    
    AI_MODEL = ml.AIModel(
        model_path=MODEL_PATH,
        tokenizer_path=TOKENIZER_PATH,
        metadata_path=METADATA_PATH
    )

    

@app.get('/') # /?q = Hello world
def index(q : Optional[str] = None):
    global AI_MODEL, MODEL_METADATA
    query = q or "hello world"
    preds_dict = AI_MODEL.predict(query)
    return {"query" : query, "results" : preds_dict}