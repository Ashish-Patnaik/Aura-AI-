# app.py
import os
import json
import re
import requests
import edge_tts
import asyncio
import nest_asyncio
import time
import gevent # Import gevent

from flask import Flask, render_template
from flask_cors import CORS
from flask_sock import Sock

nest_asyncio.apply()

# --- Configuration ---
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set! Please set it in your Render environment.")

Model = "gemini-2.0-flash"

GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{Model}:streamGenerateContent?key={GEMINI_API_KEY}&alt=sse"
FIXED_VOICE = "en-US-JennyNeural"
SYSTEM_PROMPT = """
You are "Aura," a friendly and helpful AI voice assistant. Keep your responses concise, natural, and to the point, as if you were speaking in a real conversation. Do not use markdown or formatting.
"""

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app)
sock = Sock(app)

# --- TTS Streamer (Unchanged)---
async def tts_streamer(text_chunk: str, websocket):
    try:
        communicate = edge_tts.Communicate(text=text_chunk, voice=FIXED_VOICE)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                if websocket.connected:
                    websocket.send(chunk["data"])
                else: break
    except Exception as e:
        print(f"Error during TTS generation: {e}")

# --- Helper for Gemini Request ---
def get_gemini_response(headers, payload):
    """Makes the Gemini API request. To be run in a gevent greenlet."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(GEMINI_API_URL, headers=headers, json=payload, stream=True)
            if response.status_code == 429:
                wait_time = (attempt + 1) * 2
                print(f"Rate limit hit. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}. Retrying...")
            time.sleep(2)
    return None


# --- Main WebSocket Route (Now with Worker-Safe Loop) ---
@sock.route('/stream')
def stream(ws):
    print("WebSocket connection established. Initializing conversation history.")
    history = [
        {"role": "user", "parts": [{"text": SYSTEM_PROMPT}]},
        {"role": "model", "parts": [{"text": "Okay, I'm ready to help."}]}
    ]

    while ws.connected:
        try:
            user_prompt = ws.receive(timeout=3600)
            if user_prompt is None: continue

            print(f"Received prompt: {user_prompt}")
            history.append({"role": "user", "parts": [{"text": user_prompt}]})
            headers = {'Content-Type': 'application/json'}
            payload = {"contents": history}

            # Spawn the Gemini request in a separate greenlet
            gemini_greenlet = gevent.spawn(get_gemini_response, headers, payload)
            gemini_greenlet.join() # Wait for the request to complete
            response = gemini_greenlet.value

            if response is None or not response.ok:
                print("Failed to get a successful response from Gemini.")
                if ws.connected:
                    ws.send(json.dumps({"type": "error", "message": "The AI is busy, please try again."}))
                continue
            
            # === THE KEY FIX: Re-acquire the event loop for each turn ===
            loop = asyncio.get_event_loop()
            
            text_buffer = ""
            full_model_response = ""
            sentence_end_re = re.compile(r'(.*?[.!?]["”’]?)', re.DOTALL)
            ws.send(json.dumps({"type": "start_of_response"}))

            for line in response.iter_lines():
                if not ws.connected: break
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data: '):
                        try:
                            chunk = json.loads(decoded_line[6:])
                            text_chunk = chunk["candidates"][0]["content"]["parts"][0]["text"]
                            text_buffer += text_chunk
                            full_model_response += text_chunk

                            match = sentence_end_re.search(text_buffer)
                            while match:
                                sentence = match.group(1).strip()
                                text_buffer = text_buffer[match.end():].lstrip()
                                if sentence:
                                    ws.send(json.dumps({"type": "text_chunk", "data": sentence}))
                                    # Use the fresh loop object
                                    loop.run_until_complete(tts_streamer(sentence, ws))
                                if not ws.connected: break
                                match = sentence_end_re.search(text_buffer)
                            if not ws.connected: break
                        except (json.JSONDecodeError, KeyError, IndexError):
                            pass
            
            if ws.connected and text_buffer.strip():
                remaining_text = text_buffer.strip()
                ws.send(json.dumps({"type": "text_chunk", "data": remaining_text}))
                loop.run_until_complete(tts_streamer(remaining_text, ws))

            if full_model_response.strip():
                history.append({"role": "model", "parts": [{"text": full_model_response.strip()}]})

            if ws.connected:
                ws.send(json.dumps({"type": "end_of_response"}))

        except gevent.timeout.Timeout:
            print("WebSocket timed out waiting for message.")
            break
        except Exception as e:
            print(f"An error occurred in the WebSocket loop: {e}")
            break
    
    print("WebSocket processing finished. Closing connection.")

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    print("Starting Flask server...")
    # NOTE: For production on Render, you should be using gunicorn, not app.run()
    # The start command should be: gunicorn --worker-class geventwebsocket.gunicorn.workers.GeventWebSocketWorker -w 1 app:app
    app.run(host='0.0.0.0', port=5000, debug=True)
