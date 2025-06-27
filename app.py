import streamlit as st
from transformers import pipeline
from deep_translator import GoogleTranslator
from gtts import gTTS
from datetime import datetime
import plotly.graph_objects as go

# --------- UI + Styling ---------
st.set_page_config(page_title="Emotiva", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    
    .stApp {
        background-color: #1a1a1a;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    /* Remove default Streamlit padding and margins */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        max-width: 100%;
    }
    
    /* Hide Streamlit header and footer */
    header[data-testid="stHeader"] {
        display: none;
    }
    
    .stDeployButton {
        display: none;
    }
    
    /* Remove white bars */
    .st-emotion-cache-6qob1r,
    .st-emotion-cache-1avcm0n {
        display: none !important;
    }
    
    /* Left panel styling */
    .left-panel {
        background-color: #2d2d2d;
        padding: 20px;
        text-align: center;
        border-radius: 15px;
        margin-right: 10px;
    }
    
    /* Chat container styling */
    .chat-container {
        background-color: #2d2d2d;
        border: 2px solid #404040;
        border-radius: 15px;
        padding: 20px;
        height: 500px;
        overflow-y: auto;
        margin-bottom: 10px;
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    
    /* Chat message styling */
    .user-message {
        background-color: #0066cc;
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        max-width: 80%;
        margin-left: auto;
        word-wrap: break-word;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .bot-message {
        background-color: #404040;
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        max-width: 80%;
        margin-right: auto;
        word-wrap: break-word;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Input container styling */
    .input-container {
        background-color: #1a1a1a;
        padding: 10px 0;
        border-top: none;
    }
    
    /* Custom scrollbar */
    .chat-container::-webkit-scrollbar {
        width: 8px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: #1a1a1a;
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: #666666;
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb:hover {
        background: #888888;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #0066cc;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #0052a3;
        transform: translateY(-2px);
    }
    
    /* Select box styling */
    .stSelectbox > div > div {
        background-color: #2d2d2d;
        color: white;
        border: 1px solid #404040;
    }
    
    /* Chat input styling */
    .stChatInput > div {
        background-color: #2d2d2d;
        border: 2px solid #404040;
        border-radius: 25px;
    }
    
    .stChatInput input {
        background-color: transparent;
        color: white;
        border: none;
    }
    
    /* Expander styling for mood popup */
    .streamlit-expanderHeader {
        background-color: #2d2d2d;
        color: white;
        border: 1px solid #404040;
        border-radius: 10px;
    }
    
    .streamlit-expanderContent {
        background-color: #2d2d2d;
        border: 1px solid #404040;
        border-radius: 0 0 10px 10px;
    }
    
    /* Modal-like styling for mood analysis */
    .mood-modal {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background-color: #2d2d2d;
        border: 2px solid #0066cc;
        border-radius: 15px;
        padding: 20px;
        z-index: 1000;
        max-width: 80%;
        max-height: 80%;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    }
    
    .mood-modal-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0,0,0,0.7);
        z-index: 999;
    }
    </style>
    """, unsafe_allow_html=True)

# --------- Language Configuration ---------
LANGUAGES = {
    'English': {'code': 'en', 'tts': 'en'},
    'हिन्दी': {'code': 'hi', 'tts': 'hi'},
    'Hinglish': {'code': 'en', 'tts': 'en'},
    'தமிழ்': {'code': 'ta', 'tts': 'ta'},
    'తెలుగు': {'code': 'te', 'tts': 'te'},
    'বাংলা': {'code': 'bn', 'tts': 'bn'},
    'मराठी': {'code': 'mr', 'tts': 'mr'}
}

# --------- Load Models & States ---------
@st.cache_resource
def load_model():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

classifier = load_model()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "selected_language" not in st.session_state:
    st.session_state.selected_language = 'English'

if "mood_history" not in st.session_state:
    st.session_state.mood_history = []

if "show_mood_popup" not in st.session_state:
    st.session_state.show_mood_popup = False

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

# --------- Generate Female Reply ---------
def generate_reply(user_input, emotion, language):
    user_input_lower = user_input.lower()

    if any(greet in user_input_lower for greet in ["hi", "hello", "hey", "heyy"]):
        core = {
            'English': "Hello dear! How are you doing today? I'm here to help you with anything you need!",
            'हिन्दी': "नमस्ते प्रिय! आप कैसे हैं? मैं आपकी हर बात में मदत करने के लिए यहाँ हूँ!",
            'Hinglish': "Hello dear! Aap kaise hain? Main aapki har help karne ke liye yahan hun!",
            'தமிழ்': "வணக்கம் அன்பே! நீங்கள் எப்படி இருக்கிறீர்கள்? நான் உங்களுக்கு எல்லாவற்றிலும் உதவ இங்கே இருக்கிறேன்!",
            'తెలుగు': "నమస్కారం ప్రియమైన! మీరు ఎలా ఉన్నారు? నేను మీకు అన్ని విషయాల్లో సహాయం చేయడానికి ఇక్కడ ఉన్నాను!",
            'বাংলা': "নমস্কার প্রিয়! আপনি কেমন আছেন? আমি আপনার সব কিছুতে সাহায্য করতে এখানে আছি!",
            'मराठी': "नमस्कार प्रिय! तुम्ही कसे आहात? मी तुमच्या सर्व गोष्टींमध्ये मदत करण्यासाठी इथे आहे!"
        }
    elif any(q in user_input_lower for q in ["help", "issue", "problem", "confused", "can't"]):
        core = {
            'English': "Oh sweetie, I'm here to help you through this! Please tell me more about what's troubling you, and I'll do my best to assist you!",
            'हिन्दी': "अरे प्यारे, मैं आपकी हर परेशानी में आपके साथ हूँ! कृपया अपनी समस्या बताएं, मैं पूरी कोशिश करूंगी!",
            'Hinglish': "Oh sweetie, main aapki har problem mein help karungi! Please batao kya issue hai, main puri koshish karungi!",
            'தமிழ்': "ஓ அன்பே, இதில் உங்களுக்கு உதவ நான் இங்கே இருக்கிறேன்! உங்களை தொந்தரவு செய்வது என்னவென்று சொல்லுங்கள், நான் முடிந்த உதவி செய்வேன்!",
            'తెలుగు': "ఓ చెల్లీ, ఈ విषయంలో మీకు సహాయం చేయడానికి నేను ఇక్కడ ఉన్నాను! మిమ్మల్ని ఇబ్బంది పెట్టేది ఏమిటో చెప్పండి, నేను నా వంతు సహాయం చేస్తాను!",
            'বাংলা': "ওহ সোনা, আমি এখানে আপনার সাহায্য করতে আছি! আপনার কী সমস্যা হচ্ছে তা বলুন, আমি যথাসাধ্য সাহায্য করব!",
            'मराठी': "अरे प्रिय, मी तुमच्या मदतीसाठी इथे आहे! तुम्हाला काय त्रास होत आहे ते सांगा, मी माझी पूर्ण मदत करेन!"
        }
    elif any(q in user_input_lower for q in ["order", "delivery", "refund", "status"]):
        core = {
            'English': "Of course honey! I'd be happy to help you with your order. Could you please share more details so I can assist you better?",
            'हिन्दी': "बिल्कुल प्रिय! मैं आपकी ऑर्डर में खुशी से मदत करूंगी। कृपया और जानकारी दें ताकि मैं बेहतर सहायता कर सकूं!",
            'Hinglish': "Bilkul honey! Main aapki order mein khushi se help karungi. Please aur details share karo taki main better assist kar sakun!",
            'தமிழ்': "நிச்சயமாக அன்பே! உங்கள் ஆர்டருக்கு உதவ நான் மகிழ்ச்சியாக இருக்கிறேன். மேலும் விவரங்களைப் பகிர்ந்து கொள்ளுங்கள், நான் சிறப்பாக உதவ முடியும்!",
            'తెలుగు': "అయ్యో ఖచ్చితంగా! మీ ఆర్డర్‌లో సహాయం చేయడానికి నేను సంతోషిస్తాను. మరింత వివరాలు షేర్ చేయండి, నేను మంచిగా సహాయం చేయగలను!",
            'বাংলা': "অবশ্যই প্রিয়! আপনার অর্ডারে সাহায্য করতে আমি খুশি হব। আরও বিস্তারিত জানান যাতে আমি আরও ভালো সাহায্য করতে পারি!",
            'मराठी': "नक्कीच प्रिय! तुमच्या ऑर्डरमध्ये मदत करण्यात मला आनंद होईल। अधिक तपशील द्या जेणेकरून मी चांगली मदत करू शकेन!"
        }
    else:
        core = {
            'English': "Thank you so much for your message, dear! I'm here and ready to help you with whatever you need! How can I make your day better?",
            'हिन्दी': "आपके संदेश के लिए बहुत धन्यवाद प्रिय! मैं यहाँ हूँ और आपकी हर जरूरत में मदत करने को तैयार हूँ! आपका दिन कैसे बेहतर बनाऊं?",
            'Hinglish': "Thank you so much dear! Main yahan hun aur aapki har need mein help karne ko ready hun! Aapka din kaise better banau?",
            'தமிழ்': "உங்கள் செய்திக்கு மிக்க நன்றி அன்பே! நான் இங்கே இருக்கிறேன், உங்களுக்கு தேவையான எதிலும் உதவ தயார்! உங்கள் நாளை எப்படி சிறப்பாக்கலாம்?",
            'తెలుగు': "మీ సందేశానికి చాలా ధన్యవాదాలు ప్రియमైన! నేను ఇక్కడ ఉన్నాను, మీకు అవసరమైన దేనిలోనైనా సహాయం చేయడానికి సిద్ధంగా ఉన్నాను! మీ రోజును ఎలా మెరుగుపరచాలి?",
            'বাংলা': "আপনার বার্তার জন্য অনেক ধন্যবাদ প্রিয়! আমি এখানে আছি এবং আপনার যা প্রয়োজন তাতে সাহায্য করতে প্রস্তুত! আপনার দিনটি কীভাবে আরও ভালো করতে পারি?",
            'मराठी': "तुमच्या संदेशासाठी खूप धन्यवाद प्रिय! मी इथे आहे आणि तुमची जी गरज असेल त्यामध्ये मदत करण्यासाठी तयार आहे! तुमचा दिवस कसा चांगला करू?"
        }

    empathy = {
        'happy': {
            'English': "I can feel your positive energy radiating through! ",
            'हिन्दी': "मैं आपकी सकारात्मक ऊर्जा महसूस कर सकती हूँ! ",
            'Hinglish': "Main aapki positive energy feel kar sakti hun! ",
            'தமிழ்': "உங்கள் நேர்மறை ஆற்றலை என்னால் உணர முடிகிறது! ",
            'తెలుగు': "మీ సానుకూల శక్తిని నేను అనుభవించగలను! ",
            'বাংলা': "আমি আপনার ইতিবাচক শক্তি অনুভব করতে পারছি! ",
            'मराठी': "मी तुमची सकारात्मक ऊर्जा जाणवू शकते! "
        },
        'sad': {
            'English': "I can sense you're feeling down, sweetheart. I'm here for you. ",
            'हिन्दी': "मैं महसूस कर सकती हूँ कि आप उदास हैं प्रिय। मैं आपके साथ हूँ। ",
            'Hinglish': "Main feel kar sakti hun ki aap sad hain dear. Main aapke saath hun. ",
            'தமிழ்': "நீங்கள் வருத்தமாக உணர்கிறீர்கள் என்பதை என்னால் உணர முடிகிறது அன்பே। நான் உங்களுடன் இருக்கிறேன். ",
            'తెలుగు': "మీరు దుఃఖంగా ఉన్నారని నేను అర్థం చేసుకోగలను ప్రియమైన. నేను మీతో ఉన్నాను। ",
            'বাংলা': "আমি বুঝতে পারছি আপনি মন খারাপ অনুভব করছেন প্রিয়। আমি আপনার সাথে আছি। ",
            'मराठी': "मी समजू शकते की तुम्हाला वाईट वाटत आहे प্রিय়। मी तुमच्या সোबत आहे। "
        },
        'angry': {
            'English': "I can feel your frustration, dear. Let me help you work through this gently. ",
            'हिन्दी': "मैं आपकी परेशानी समझ सकती हूँ प्रिय। आइए इसे धीरे-धीरे हल करते हैं। ",
            'Hinglish': "Main aapki frustration samajh sakti hun dear. Chaliye ise gently solve karte hain. ",
            'தமிழ்': "உங்கள் கோபத்தை என்னால் உணர முடிகிறது அன்பே। இதை மெதுவாக தீர்க்க உதவுகிறேன். ",
            'తెలుగు': "మీ నిరాశను నేను అర్థం చేసుకోగలను ప్రియమైన. దీన్ని మెల్లగా పరిష్కరించడంలో సహాయం చేస్తాను। ",
            'বাংলা': "আমি আপনার হতাশা বুঝতে পারছি প্রিয়। আসুন এটি ধীরে ধীরে সমাধান করি। ",
            'मराठी': "मी तुमची नाराजी समजू शकते প्রিয়। चला याचे हळूवारपणे निराकरण करूया। "
        },
        'fear': {
            'English': "I can sense your worry, honey. Don't be afraid, I'm here to guide you step by step. ",
            'हिन्दी': "मैं आपकी चिंता समझ सकती हूँ प्रिय। डरिए मत, मैं आपको कदम-कदम पर मार्गदर्शन दूंगी। ",
            'Hinglish': "Main aapki worry samajh sakti hun honey. Dare mat, main step by step guide karungi. ",
            'தமிழ்': "உங்கள் கவலையை என்னால் உணர முடிகிறது அன்பே। பயப்பட வேண்டாம், நான் படிப்படியாக வழிகாட்டுகிறேன். ",
            'తెలుగు': "మీ ఆందోళనను నేను అర్థం చేసుకోగలను ప్రియమైన. భయపడకండి, నేను మిమ్మల్ని దశలవారీగా మార్గనిర్దేశం చేస్తాను। ",
            'বাংলা': "আমি আপনার চিন্তা বুঝতে পারছি প্রিয়। ভয় পাবেন না, আমি আপনাকে ধাপে ধাপে গাইড করব। ",
            'मराठी': "मী तुमची चिंता समजू शकते প্রিয়। घाबरू नका, मी तुम्हाला टप्प्या टप्प्याने मार्गदर्शन करेन। "
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

# --------- Mood Analysis Chart ---------
def create_mood_chart():
    if not st.session_state.mood_history:
        return None
    
    mood_data = {
        'Message': [f"Message {i+1}" for i in range(len(st.session_state.mood_history))],
        'Emotion': st.session_state.mood_history,
        'Time': list(range(1, len(st.session_state.mood_history) + 1))
    }
    
    color_map = {
        'happy': '#4CAF50',
        'sad': '#2196F3', 
        'angry': '#F44336',
        'fear': '#FF9800',
        'surprise': '#9C27B0',
        'disgust': '#795548',
        'joy': '#FFEB3B'
    }
    
    colors = [color_map.get(emotion, '#607D8B') for emotion in mood_data['Emotion']]
    
    fig = go.Figure(data=go.Scatter(
        x=mood_data['Time'],
        y=mood_data['Emotion'],
        mode='markers+lines',
        marker=dict(size=12, color=colors, line=dict(width=2, color='white')),
        line=dict(width=3, color='rgba(50, 50, 50, 0.8)'),
        text=mood_data['Message'],
        hovertemplate='<b>%{text}</b><br>Emotion: %{y}<br><extra></extra>'
    ))
    
    fig.update_layout(
        title='Customer Mood Journey',
        xaxis_title='Message Number',
        yaxis_title='Detected Emotion',
        height=400,
        showlegend=False,
        paper_bgcolor='#2d2d2d',
        plot_bgcolor='#2d2d2d',
        font=dict(color='white')
    )
    
    return fig

# --------- Main Layout ---------
col1, col2 = st.columns([1, 2])

# Left Panel - Avatar, Title, and Buttons
with col1:
    st.markdown('<div class="left-panel">', unsafe_allow_html=True)
    
    try:
        st.image("emotiva.png", width=300)
    except:
        st.markdown("""
            <div style="text-align: center;">
                <div style="width: 300px; height: 300px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                           border-radius: 20px; margin: 0 auto; display: flex; align-items: center; justify-content: center;">
                    <span style="font-size: 120px; color: white;">🤖</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<h1 style='text-align:center; color: white;'>🤖 Emotiva</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center; color: #cccccc;'>Where Every Matter Matters</h4>", unsafe_allow_html=True)
    
    if st.button("📊 Mood Analysis", use_container_width=True):
        st.session_state.show_mood_popup = True
    
    if st.button("🔄 Start New Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.mood_history = []
        st.session_state.show_mood_popup = False
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Right Panel - Chat Interface
with col2:
    # Language selector
    col_empty, col_lang = st.columns([3, 1])
    with col_lang:
        st.session_state.selected_language = st.selectbox(
            "Language:",
            options=list(LANGUAGES.keys()),
            index=list(LANGUAGES.keys()).index(st.session_state.selected_language)
        )

    # Show mood analysis popup as modal
    if st.session_state.show_mood_popup:
        if st.session_state.mood_history:
            with st.container():
                st.markdown("---")
                st.markdown("### 📊 Customer Mood Analysis")
                st.plotly_chart(create_mood_chart(), use_container_width=True)
                
                col_close1, col_close2, col_close3 = st.columns([1, 1, 1])
                with col_close2:
                    if st.button("Close Analysis", use_container_width=True):
                        st.session_state.show_mood_popup = False
                        st.rerun()
                st.markdown("---")
        else:
            st.info("No mood data available yet. Start chatting to see mood analysis!")
            st.session_state.show_mood_popup = False

    # Chat container with better styling
    chat_html = '<div class="chat-container">'
    
    for idx, (role, message, lang) in enumerate(st.session_state.chat_history):
        if role == "user":
            chat_html += f'<div class="user-message">{message}</div>'
        else:
            chat_html += f'<div class="bot-message">{message}</div>'
    
    chat_html += '</div>'
    
    # Display chat container
    st.markdown(chat_html, unsafe_allow_html=True)

    # Input container (removed the white bar styling)
    user_input = st.chat_input("Type your message here...")

    # TTS for latest bot message only (moved to bottom)
    if st.session_state.chat_history and st.session_state.chat_history[-1][0] == "bot":
        latest_message = st.session_state.chat_history[-1][1]
        if st.button("🔊 Speak", key="latest_tts"):
            tts_lang = LANGUAGES[st.session_state.selected_language]['tts']
            audio_path = speak(latest_message, tts_lang)
            st.audio(audio_path, format="audio/mp3")

    # Handle user input
    if user_input:
        # Add user message to chat
        st.session_state.chat_history.append(("user", user_input, ''))

        # Detect emotion and language
        lang = detect_language(user_input)
        emotion = detect_emotion(user_input)
        st.session_state.mood_history.append(emotion)
        
        # Generate reply
        target_language = st.session_state.selected_language
        reply = generate_reply(user_input, emotion, target_language)
        st.session_state.chat_history.append(("bot", reply, target_language))
        
        st.rerun()