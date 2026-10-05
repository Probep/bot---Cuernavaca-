from flask import Flask, request
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = Flask(__name__)

MODEL_PATH = "distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
model.eval()

def responder(mensaje):
    prompt = f"Cliente: {mensaje}\nTu (de Cuernavaca, experto en bots de WhatsApp, respuesta corta y vendedora):"
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(inputs, max_length=inputs.shape[1]+60, temperature=0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=True).split("Tu")[-1].strip().replace("Cliente:", "").strip()

@app.route("/whatsapp", methods=["POST"])
def wa():
    msg = request.values.get("Body") or request.values.get("message") or request.json.get("message") if request.json else "Hola"
    resp = responder(msg)
    return f'<?xml version="1.0" encoding="UTF-8"?><Response><Message>{resp}</Message></Response>', 200, {'Content-Type': 'text/xml'}

@app.route("/")
def home():
    return "Bot Cuernavaca ON! Ve a /whatsapp"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port) 
