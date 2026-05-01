from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import MarianMTModel, MarianTokenizer
import os
import torch
import logging
import json
import sqlite3
import io
import base64
import requests
from vosk import Model, KaldiRecognizer
from rapidfuzz import fuzz
from datetime import datetime
from pydub import AudioSegment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates')
CORS(app)

# --- CONFIGURATION ---
MODEL_NAME = "Helsinki-NLP/opus-mt-en-ceb"
SAVE_PATH = "./cebuano_ai_model"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Vosk & DB Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model")
DB_PATH = os.path.join(BASE_DIR, "Cebuano.db")
SAMPLE_RATE = 16000

# API Configuration (Optional: Get key from https://dictionaryapi.com/)
MW_API_KEY = "YOUR_MERRIAM_WEBSTER_KEY_HERE"

# --- 1. DATA LAYER (Verified Dictionary) ---
DICTIONARY_DATA = {
    "eyes": {
        "trans": "mata",
        "type": "Noun",
        "meaning": "The organ of sight; used to see and perceive the world around you",
        "synonyms": "panan-aw, sud-ong",
        "example": "Importante ang mata aron makita ang kanindot sa kinaiyahan."
    },
    "beautiful": {
        "trans": "anindot",
        "type": "Adjective",
        "meaning": "Pleasing to the eye; attractive and aesthetically appealing",
        "synonyms": "nindot, maanyag",
        "example": "Anindot kaayo ang mga buwak sa imong tanaman."
    },
    # ... add your other manual entries here
}


# --- 2. ENGINE UTILS ---

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS test_results
                        (
                            id
                            INTEGER
                            PRIMARY
                            KEY
                            AUTOINCREMENT,
                            user_name
                            TEXT,
                            target_word
                            TEXT,
                            spoken_text
                            TEXT,
                            accuracy
                            REAL,
                            passed
                            INTEGER,
                            timestamp
                            DATETIME
                        )''')


def expand_contractions(text):
    contractions = {"can't": "cannot", "don't": "do not", "it's": "it is", "i'm": "i am"}
    for word, replacement in contractions.items():
        text = text.replace(word, replacement)
    return text


def get_word_details(word):
    """Fetches details from 3 APIs: FreeDict, Datamuse, and Merriam-Webster."""
    details = {"meaning": "N/A", "synonyms": "N/A", "example": "N/A"}

    # API 1: Free Dictionary API
    try:
        dict_res = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}", timeout=3)
        if dict_res.status_code == 200:
            data = dict_res.json()[0]
            details['meaning'] = data['meanings'][0]['definitions'][0].get('definition', "N/A")
            details['example'] = data['meanings'][0]['definitions'][0].get('example', "N/A")
            syns = data['meanings'][0].get('synonyms', [])
            if syns: details['synonyms'] = ", ".join(syns[:3])
            return details
    except:
        pass

    # API 2: Datamuse (Fallback for Synonyms)
    try:
        dm_res = requests.get(f"https://api.datamuse.com/words?rel_syn={word}&max=3", timeout=3)
        if dm_res.status_code == 200:
            syns = [item['word'] for item in dm_res.json()]
            if syns: details['synonyms'] = ", ".join(syns)
    except:
        pass

    # API 3: Merriam-Webster (Fallback for Meaning)
    if details['meaning'] == "N/A" and MW_API_KEY != "YOUR_MERRIAM_WEBSTER_KEY_HERE":
        try:
            mw_res = requests.get(
                f"https://www.dictionaryapi.com/api/v3/references/collegiate/json/{word}?key={MW_API_KEY}", timeout=3)
            if mw_res.status_code == 200:
                data = mw_res.json()
                if data and isinstance(data, list):
                    details['meaning'] = data[0].get('shortdef', ["N/A"])[0]
        except:
            pass

    return details


def neural_translate(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)
    translated_tokens = translation_model.generate(**inputs, num_beams=8, max_length=128)
    return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)


def process_audio(audio_b64):
    try:
        if "," in audio_b64: audio_b64 = audio_b64.split(",")[1]
        audio_data = base64.b64decode(audio_b64)
        audio = AudioSegment.from_file(io.BytesIO(audio_data))
        audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1)
        return audio.raw_data
    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        return None


# --- 3. ROUTES ---

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/translate', methods=['POST'])
def translate_text():
    try:
        data = request.get_json()
        user_input = data.get("text", "").strip().lower()
        if not user_input:
            return jsonify({"translated": "", "explanation": "Palihog pagbutang og pulong."})

        clean_text = expand_contractions(user_input)

        # Step 1: Check Manual Dictionary
        if user_input in DICTIONARY_DATA:
            result = DICTIONARY_DATA[user_input]
            source = "Verified Dictionary"
        else:
            # Step 2 & 3: AI Translation + API Details
            ai_trans = neural_translate(clean_text)
            api_data = get_word_details(user_input)
            result = {
                "trans": ai_trans,
                "type": "AI & API Enhanced",
                "meaning": api_data['meaning'],
                "synonyms": api_data['synonyms'],
                "example": api_data['example']
            }
            source = "Neural AI + External APIs"

        # Format Explanation in Cebuano
        explanation = (
            f"Ang pulong nga '{user_input}' usa ka {result.get('type', 'pulong').lower()}. "
            f"Sa Cebuano, kini nagpasabut nga '**{result['trans']}**'. "
            f"Kini nagpasabut: {result['meaning']}. "
            f"Pananglitan: '*{result['example']}*'"
        )

        return jsonify({
            "translated": result['trans'].capitalize(),
            "explanation": explanation,
            "details": result,
            "meta": {"source": source}
        }), 200
    except Exception as e:
        logger.error(f"Translation Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/rate_pronunciation', methods=['POST'])
def rate_pronunciation():
    # ... (Keep your existing Vosk pronunciation logic here)
    pass


if __name__ == '__main__':
    init_db()
    # Initializing AI Models
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    translation_model = MarianMTModel.from_pretrained(MODEL_NAME).to(device)
    vosk_model = Model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

    app.run(host='0.0.0.0', port=5000, threaded=True)