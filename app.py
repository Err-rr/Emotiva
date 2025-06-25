import streamlit as st
from transformers import pipeline
from deep_translator import GoogleTranslator
from gtts import gTTS
from datetime import datetime

# --------- UI + Styling ---------
st.set_page_config(page_title="Emotiva", layout="centered")

st.markdown("""
    <style>
    body {
        background-color: #fffaf0;
    }
    .main {
        background-color: #fffaf0;
    }
    h1, h4 {
        color: #0d47a1;
    }
    .st-emotion-cache-1v0mbdj, .st-emotion-cache-1avcm0n {
        color: black !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Display avatar and title
st.image("emotiva.png", width=150)
st.markdown("<h1 style='text-align:center;'>🤖 Emotiva</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>Where Every Mood Matters</h4>", unsafe_allow_html=True)
st.divider()

# --------- Load Models & States ---------
@st.cache_resource
def load_model():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

classifier = load_model()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --------- Language Detection ---------
def detect_language(text):
    try:
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        return 'hi' if translated != text else 'en'
    except:
        return 'en'

# --------- Emotion Detection ---------
def detect_emotion(text):
    result = classifier(text)[0]
    top = max(result, key=lambda x: x['score'])
    return top['label'].lower()

# --------- Generate Reply ---------
def generate_reply(user_input, emotion, language):
    user_input_lower = user_input.lower()

    if any(greet in user_input_lower for greet in ["hi", "hello", "hey", "heyy"]):
        core = {
            'en': "Hey there! How are you doing today? How can I help you?",
            'hi': "नमस्ते! आप कैसे हैं? मैं आपकी कैसे मदद कर सकता हूँ?"
        }
    elif any(q in user_input_lower for q in ["help", "issue", "problem", "confused", "can't"]):
        core = {
            'en': "I'm here to help you. Can you please describe your issue?",
            'hi': "मैं आपकी मदद करने के लिए यहाँ हूँ। कृपया अपनी समस्या बताएं।"
        }
    elif any(q in user_input_lower for q in ["order", "delivery", "refund", "status"]):
        core = {
            'en': "Let me help you with your order. Can you give me more details?",
            'hi': "चलिए मैं आपकी ऑर्डर से जुड़ी मदद करता हूँ। कृपया और जानकारी दें।"
        }
    else:
        core = {
            'en': "Thanks for your message! How can I assist you further?",
            'hi': "आपके संदेश के लिए धन्यवाद! मैं और कैसे मदद कर सकता हूँ?"
        }

    empathy = {
        'happy': {
            'en': "You seem cheerful! 😊 ",
            'hi': "आप खुश लग रहे हैं! 😊 "
        },
        'sad': {
            'en': "I'm here for you. ",
            'hi': "मैं आपके साथ हूँ। "
        },
        'angry': {
            'en': "I'm really sorry you're upset. ",
            'hi': "हमें खेद है कि आप नाराज़ हैं। "
        },
        'confused': {
            'en': "Let me help clear things up. ",
            'hi': "आइए मैं आपकी उलझन दूर करता हूँ। "
        }
    }

    e_text = empathy.get(emotion, {}).get(language, "")
    return f"{e_text}{core[language]}"

# --------- TTS ---------
def speak(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    filename = f"response_{datetime.now().timestamp()}.mp3"
    tts.save(filename)
    return filename

# --------- Chat Display ---------
for idx, (role, message, lang) in enumerate(st.session_state.chat_history):
    if role == "user":
        st.chat_message("user").write(message)
    else:
        with st.chat_message("assistant"):
            st.write(message)
            if st.button("🔊 Speak", key=f"tts_{idx}"):
                audio_path = speak(message, lang)
                st.audio(audio_path, format="audio/mp3")

# --------- New User Message ---------
user_input = st.chat_input("Type your message here...")

if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.chat_history.append(("user", user_input, ''))

    lang = detect_language(user_input)
    emotion = detect_emotion(user_input)
    reply = generate_reply(user_input, emotion, lang)
    st.session_state.chat_history.append(("bot", reply, lang))

    with st.chat_message("assistant"):
        st.write(reply)
        if st.button("🔊 Speak", key=f"tts_new"):
            audio_path = speak(reply, lang)
            st.audio(audio_path, format="audio/mp3")
