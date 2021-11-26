import pathlib
import os
from cryptography.fernet import Fernet
from dotenv import load_dotenv

load_dotenv()

def generate_key():
    return Fernet.generate_key().decode('UTF-8')


ENCRYPTION_KEY = os.environ.get('ENCRYPTION_KEY')
BASE_DIR = pathlib.Path().resolve().parent
BASE_DIR.exists()


APP_DIR = BASE_DIR / 'app'
APP_DIR.exists()

IGNORED_DIR = APP_DIR / 'ignored'
SECURED_DIR = APP_DIR / 'encrypted'
DECRYPTED_DIR = APP_DIR / 'decrypted'


def encrypt_dir(input_dir, output_dir):
    key = ENCRYPTION_KEY
    if not key:
        raise Exception("Encryption Key not found!")
    fer = Fernet(key)
    input_dir = pathlib.Path(input_dir)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    for path in input_dir.glob("*"):
        _path_bytes = path.read_bytes() #open("filepath", 'rb')
        data = fer.encrypt(_path_bytes)
        rel_path = path.relative_to(input_dir) 
        dest_path = output_dir / rel_path
        dest_path.write_bytes(data)


def decrypt_dir(input_dir, output_dir):
    key = ENCRYPTION_KEY
    if not key:
        raise Exception("Encryption Key not found!")
    fer = Fernet(key)
    input_dir = pathlib.Path(input_dir)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    for path in input_dir.glob("*"):
        _path_bytes = path.read_bytes()
        data = fer.decrypt(_path_bytes)
        rel_path = path.relative_to(input_dir)
        dest_path = output_dir / rel_path
        dest_path.write_bytes(data)



encrypt_dir(str(IGNORED_DIR), str(SECURED_DIR))

decrypt_dir(str(SECURED_DIR), str(DECRYPTED_DIR))





