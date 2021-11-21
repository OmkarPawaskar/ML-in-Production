import pathlib
import json
from typing import Optional
from fastapi import FastAPI
from cassandra.cqlengine.management import sync_table
from . import (
    config,
    db,
    models,
    ml
)


app = FastAPI()
settings = config.get_settings()
BASE_DIR = pathlib.Path(__file__).resolve().parent #fastapi base dir

MODEL_DIR = BASE_DIR.parent / "models" 
SMS_SPAM_MODEL_PATH = MODEL_DIR / "spam-sms"
MODEL_PATH = SMS_SPAM_MODEL_PATH / "spam-model.h5"
TOKENIZER_PATH = SMS_SPAM_MODEL_PATH / "spam-classifer-tokenizer.json"
METADATA_PATH = SMS_SPAM_MODEL_PATH / "spam-classifer-metadata.json"

AI_MODEL = None
DB_SESSION = None
SMS_INFERENCE = models.SMSInference




@app.on_event("startup")
def on_startup():
    global AI_MODEL, DB_SESSION
    #load my model
    
    AI_MODEL = ml.AIModel(
        model_path=MODEL_PATH,
        tokenizer_path=TOKENIZER_PATH,
        metadata_path=METADATA_PATH
    )
    DB_SESSION = db.get_session()
    sync_table(SMS_INFERENCE) #To sync the models to the database
    

@app.get('/') # /?q = Hello world
def index(q : Optional[str] = None):
    global AI_MODEL
    query = q or "hello world"
    preds_dict = AI_MODEL.predict(query)
    return {"query" : query, "results" : preds_dict}