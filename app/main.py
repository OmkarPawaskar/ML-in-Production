import pathlib
from fastapi import FastAPI

app = FastAPI()
BASE_DIR = pathlib.Path(__file__).resolve().parent #fastapi base dir

MODEL_DIR = BASE_DIR.parent / "models" 
SMS_SPAM_MODEL_PATH = MODEL_DIR / "spam-sms"
MODEL_PATH = SMS_SPAM_MODEL_PATH / "spam-model.h5"
TOKENIZER_PATH = SMS_SPAM_MODEL_PATH / "spam-classifer-tokenizer.json"
METADATA_PATH = SMS_SPAM_MODEL_PATH / "spam-classifer-metadata.json"

@app.get('/')
def index():
    return {"hello": "world", "BASE_DIR": BASE_DIR, "MODEL_DIR" : MODEL_DIR.exists(), "MODEL_PATH" : MODEL_PATH.exists()} 