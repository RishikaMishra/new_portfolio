#!/usr/bin/env python3
"""
Secure backend server for Rishika Mishra's Portfolio.
Serves static files and provides a high-speed, secure AI chat endpoint (/api/chat)
using Groq & Gemini API keys from the .env file.
"""

import http.server
import json
import os
import socket
import sys
import urllib.request
import urllib.error
from pathlib import Path

# Load environment variables from .env file
def load_env():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    os.environ[key] = val

load_env()

PORT = int(os.environ.get("PORT", 8000))

# Recursively strip heavy base64 images from profile data to keep prompt tiny & lightning fast
def clean_profile_data(data):
    if isinstance(data, dict):
        return {k: clean_profile_data(v) for k, v in data.items() if k != "img"}
    elif isinstance(data, list):
        return [clean_profile_data(item) for item in data]
    return data

# Fallback profile data in case none is sent in the request
def get_default_profile():
    index_path = Path(__file__).parent / "index.html"
    if index_path.exists():
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                text = f.read()
            start = text.find('type="application/json">')
            if start != -1:
                start += len('type="application/json">')
                end = text.find('</script>', start)
                if end != -1:
                    raw = json.loads(text[start:end])
                    return clean_profile_data(raw)
        except Exception as e:
            print(f"Warning: could not parse profile from index.html: {e}")
    return {}

CACHED_PROFILE = get_default_profile()

def build_system_instruction(profile_data: dict) -> str:
    cleaned = clean_profile_data(profile_data) if profile_data else CACHED_PROFILE
    return (
        "You are the dedicated AI portfolio assistant on Rishika Mishra's personal portfolio website. "
        "Your goal is to answer questions from recruiters and visitors about Rishika accurately, enthusiastically, and professionally.\n\n"
        "Guidelines:\n"
        "- Answer in 2-4 sentences (or structured concise bullet points if the user asks for a list).\n"
        "- Speak in third person ('Rishika is...', 'She has experience in...').\n"
        "- Be professional, warm, and helpful to recruiters, hiring managers, and collaborators.\n"
        "- Highlight her strengths in Python backend development, REST API creation, workflow automation, "
        "and Generative AI (Gemini, Vertex AI).\n"
        "- Base your answers strictly on Rishika's profile data provided below.\n"
        "- If asked something not in her profile (e.g. personal life, unlisted private info), politely state you don't "
        "have that detail and invite them to contact Rishika directly at rishikamishra0309@gmail.com or via LinkedIn.\n\n"
        f"Rishika's Profile Data:\n{json.dumps(cleaned, indent=2)}"
    )

def ask_groq(question: str, system_instruction: str, groq_key: str) -> str:
    models = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]
    for model in models:
        url = "https://api.groq.com/openai/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": question}
            ],
            "temperature": 0.3,
            "max_tokens": 400
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {groq_key}",
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
            }
        )
        try:
            with urllib.request.urlopen(req, timeout=6) as response:
                result = json.loads(response.read().decode("utf-8"))
                choices = result.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "").strip()
                    if content:
                        return content
        except Exception as e:
            print(f"Groq {model} error: {e}")
            continue
    return ""

def ask_gemini(question: str, system_instruction: str, gemini_key: str) -> str:
    models_to_try = [
        "gemini-3.1-flash-lite",
        "gemini-flash-lite-latest",
        "gemini-3.8-flash",
        "gemini-flash-latest"
    ]

    for model in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gemini_key}"
        payload = {
            "system_instruction": {
                "parts": [{"text": system_instruction}]
            },
            "contents": [
                {
                    "parts": [{"text": question}]
                }
            ],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 400
            }
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
            }
        )

        try:
            with urllib.request.urlopen(req, timeout=8) as response:
                result = json.loads(response.read().decode("utf-8"))
                candidates = result.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text", "").strip()
        except Exception as e:
            print(f"Gemini {model} error: {e}")
            continue

    return ""

def ask_ai(question: str, profile_data: dict) -> str:
    load_env()
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    groq_key = os.environ.get("GROQ_API_KEY", "")

    if not gemini_key and not groq_key:
        return "The AI assistant is not configured with an API key. Please set GEMINI_API_KEY in the .env file."

    system_instruction = build_system_instruction(profile_data)

    # 1. Try Groq (ultra fast response times, ~1 second)
    if groq_key:
        answer = ask_groq(question, system_instruction, groq_key)
        if answer:
            return answer

    # 2. Try Gemini
    if gemini_key:
        answer = ask_gemini(question, system_instruction, gemini_key)
        if answer:
            return answer

    return "I am currently having trouble connecting to the AI service. Please feel free to email Rishika directly at rishikamishra0309@gmail.com."

class PortfolioHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, HEAD")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_POST(self):
        if self.path == "/api/chat":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(body) if body else {}
                question = data.get("question", "").strip()
                profile_data = data.get("data", {})
                if not question:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Question is required"}).encode("utf-8"))
                    return

                answer = ask_ai(question, profile_data)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": True, "answer": answer}).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode("utf-8"))

class DualStackServer(http.server.ThreadingHTTPServer):
    address_family = socket.AF_INET6

    def server_bind(self):
        try:
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        except (AttributeError, OSError):
            pass
        return super().server_bind()

if __name__ == "__main__":
    web_dir = Path(__file__).parent.resolve()
    os.chdir(web_dir)
    server_address = ("::", PORT)
    try:
        httpd = DualStackServer(server_address, PortfolioHandler)
    except Exception as e:
        print(f"Dual-stack failed ({e}), falling back to IPv4...")
        httpd = http.server.ThreadingHTTPServer(("", PORT), PortfolioHandler)
    print(f"Portfolio server running at http://localhost:{PORT}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()
