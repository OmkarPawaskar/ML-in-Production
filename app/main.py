import pathlib
import json
from typing import Optional
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from cassandra.query import SimpleStatement
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
    top = preds_dict.get('top')
    data = {"query" : query, **top}
    obj = SMS_INFERENCE.objects.create(**data)
    return obj

@app.get('/inferences/')
def list_inference():
    obj = SMS_INFERENCE.objects.all()
    #print(obj)
    return list(obj)

@app.get('/inferences/{my_uuid}')
def read_inference(my_uuid):
    obj = SMS_INFERENCE.objects.get(uuid=my_uuid)
    return obj

def fetch_rows(
    stmt : SimpleStatement,
    fetch_size : int = 25,
    session = None
    ):
    stmt.fetch_size = fetch_size
    result_set = session.execute(stmt)
    has_pages = result_set.has_more_pages
    while has_pages:
        for row in result_set.current_rows:
            yield f"{row['uuid']},{row['label']},{row['confidence']},{row['query']},{row['model_version']}\n"
        has_pages = result_set.has_more_pages #will return False in end if there are no records left
        result_set = session.execute(stmt, paging_state=result_set.paging_state) #uses paging state to get remaining records

@app.get('/dataset')
def export_inferences():
    global DB_SESSION
    cql_query = "SELECT * FROM spam_inferences.smsinference LIMIT 10000"
    statement = SimpleStatement(cql_query)
    #rows = DB_SESSION.execute(cql_query)
    return StreamingResponse(fetch_rows(statement, 25, DB_SESSION))