"""
styles.py — Custom CSS for MoodBite Streamlit App

Aesthetic direction: Warm editorial / organic restaurant menu
- Deep terracotta + warm cream palette
- Playfair Display (serif display) + DM Sans (body)
- Card-based food recommendations with hover lift
- Conversational chat bubbles with personality
- Mood pills with distinct color coding per emotion
"""

MAIN_CSS = """
<style>
/* ============================================================
   GOOGLE FONTS
   ============================================================ */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400;1,600&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap');

/* ============================================================
   CSS VARIABLES — Design Tokens
   ============================================================ */
:root {
    /* Core palette */
    --cream:        #FAF6F0;
    --cream-dark:   #F2EBE0;
    --terracotta:   #C4572A;
    --terracotta-lt:#E07A52;
    --brown-dark:   #2C1810;
    --brown-mid:    #5C3D2E;
    --brown-lt:     #8B6655;
    --gold:         #D4A853;
    --gold-lt:      #EDD28A;
    --sage:         #7A9E7E;
    --sage-lt:      #A8C5AC;
    --dusty-rose:   #C48B8B;
    --dusty-rose-lt:#E0B8B8;
    --slate:        #6B7E8C;
    --slate-lt:     #A8BAC4;

    /* Semantic */
    --bg-main:      var(--cream);
    --bg-card:      #FFFFFF;
    --bg-sidebar:   #2C1810;
    --text-primary: var(--brown-dark);
    --text-secondary: var(--brown-lt);
    --text-muted:   #A09080;
    --accent:       var(--terracotta);
    --accent-hover: var(--terracotta-lt);
    --border:       #E8DDD0;
    --shadow-sm:    0 2px 8px rgba(44,24,16,0.06);
    --shadow-md:    0 6px 24px rgba(44,24,16,0.10);
    --shadow-lg:    0 16px 48px rgba(44,24,16,0.14);

    /* Typography */
    --font-display: 'Playfair Display', Georgia, serif;
    --font-body:    'DM Sans', system-ui, sans-serif;

    /* Spacing */
    --radius-sm:    8px;
    --radius-md:    16px;
    --radius-lg:    24px;
    --radius-pill:  100px;
}

/* ============================================================
   GLOBAL RESET & BASE
   ============================================================ */
html, body, [class*="css"] {
    font-family: var(--font-body) !important;
    color: var(--text-primary) !important;
    background-color: var(--bg-main) !important;
}

/* Remove default Streamlit padding */
.main .block-container {
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
    max-width: 900px !important;
}

/* Hide Streamlit default header & footer */
#MainMenu, footer, header { visibility: hidden !important; }

/* ============================================================
   SIDEBAR
   ============================================================ */
[data-testid="stSidebar"] {
    background-color: var(--bg-sidebar) !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}

[data-testid="stSidebar"] * {
    color: var(--cream) !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: var(--font-display) !important;
    color: var(--gold-lt) !important;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.12) !important;
}

[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label {
    color: var(--cream-dark) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

/* ============================================================
   APP HEADER
   ============================================================ */
.moodbite-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    position: relative;
}

.moodbite-header::after {
    content: '';
    display: block;
    width: 64px;
    height: 3px;
    background: linear-gradient(90deg, var(--terracotta), var(--gold));
    margin: 1.2rem auto 0;
    border-radius: var(--radius-pill);
}

.moodbite-logo {
    font-family: var(--font-display) !important;
    font-size: 3rem !important;
    font-weight: 700 !important;
    color: var(--brown-dark) !important;
    letter-spacing: -0.02em;
    line-height: 1;
    margin: 0 !important;
}

.moodbite-logo span {
    color: var(--terracotta);
}

.moodbite-tagline {
    font-family: var(--font-display) !important;
    font-style: italic;
    font-size: 1.05rem !important;
    color: var(--text-secondary) !important;
    margin-top: 0.4rem !important;
    font-weight: 400 !important;
}

/* ============================================================
   MOOD SELECTOR
   ============================================================ */
.mood-section-title {
    font-family: var(--font-body);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
    margin-top: 2rem;
    text-align: center;
}

.mood-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    justify-content: center;
    margin-bottom: 1.5rem;
}

.mood-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.45rem 1rem;
    border-radius: var(--radius-pill);
    font-family: var(--font-body);
    font-size: 0.88rem;
    font-weight: 500;
    cursor: pointer;
    border: 1.5px solid transparent;
    transition: all 0.2s ease;
    user-select: none;
    white-space: nowrap;
}

.mood-pill:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}

.mood-pill.active {
    border-color: currentColor;
    box-shadow: var(--shadow-md);
    font-weight: 600;
}

/* Per-mood colour coding */
.mood-happy    { background: #FFF8E7; color: #B8860B; }
.mood-happy.active { background: #FFF0C0; }
.mood-sad      { background: #EEF4FF; color: #4A6FA5; }
.mood-sad.active { background: #D8E8FF; }
.mood-stressed { background: #FFF0F0; color: #C0392B; }
.mood-stressed.active { background: #FFD8D8; }
.mood-tired    { background: #F0EEF8; color: #7B68EE; }
.mood-tired.active { background: #E0DCF8; }
.mood-romantic { background: #FFF0F5; color: #C2185B; }
.mood-romantic.active { background: #FFD8E8; }
.mood-excited  { background: #FFF3E8; color: #E65100; }
.mood-excited.active { background: #FFE0C0; }
.mood-cozy     { background: #F5F0E8; color: #795548; }
.mood-cozy.active { background: #EDE0C8; }
.mood-adventurous { background: #EDFAF4; color: #2E7D52; }
.mood-adventurous.active { background: #C8F0DC; }
.mood-anxious      { background: #F0F4FF; color: #3A5BA0; }
.mood-anxious.active { background: #D0DEFF; }
.mood-bored        { background: #F5F5F0; color: #7A7A6A; }
.mood-bored.active { background: #E8E8DC; }
.mood-nostalgic    { background: #FAF0E6; color: #8B5E3C; }
.mood-nostalgic.active { background: #F0DCC8; }
.mood-celebratory  { background: #FFF9E6; color: #B8860B; }
.mood-celebratory.active { background: #FFF0C0; }
.mood-lonely       { background: #F0F0FA; color: #5A5A9A; }
.mood-lonely.active { background: #DCDCF8; }
.mood-energetic    { background: #FFFBE6; color: #B8780A; }
.mood-energetic.active { background: #FFF0B0; }
.mood-sluggish     { background: #EFF8F0; color: #2E7A42; }
.mood-sluggish.active { background: #C8ECD0; }
.mood-focused      { background: #EDF5FF; color: #1A5EA8; }
.mood-focused.active { background: #C8DCFF; }
.mood-heartbroken  { background: #FFF0F8; color: #C2185B; }
.mood-heartbroken.active { background: #FFD0E8; }
.mood-proud        { background: #FFFBEC; color: #B07A00; }
.mood-proud.active { background: #FFF0C0; }
.mood-nervous      { background: #F5F5FA; color: #6A6A9A; }
.mood-nervous.active { background: #E0E0F0; }
.mood-content      { background: #F0FAF0; color: #2E7A42; }
.mood-content.active { background: #C8ECD0; }

/* ============================================================
   CHAT CONTAINER
   ============================================================ */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 1.25rem;
    padding: 1.5rem 0;
}

/* Divider between mood selector and chat */
.chat-divider {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 1rem 0;
    color: var(--text-muted);
    font-size: 0.78rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.chat-divider::before,
.chat-divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ============================================================
   CHAT BUBBLES
   ============================================================ */
.chat-message {
    display: flex;
    flex-direction: column;
    max-width: 82%;
    animation: slideIn 0.3s ease;
}

@keyframes slideIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

.chat-message.user {
    align-self: flex-end;
    align-items: flex-end;
}

.chat-message.assistant {
    align-self: flex-start;
    align-items: flex-start;
}

.chat-bubble {
    padding: 0.85rem 1.1rem;
    border-radius: var(--radius-md);
    font-size: 0.95rem;
    line-height: 1.6;
    position: relative;
}

.chat-bubble.user-bubble {
    background: var(--brown-dark);
    color: var(--cream) !important;
    border-bottom-right-radius: 4px;
    box-shadow: var(--shadow-sm);
}

.chat-bubble.assistant-bubble {
    background: var(--bg-card);
    color: var(--text-primary) !important;
    border-bottom-left-radius: 4px;
    border: 1px solid var(--border);
    box-shadow: var(--shadow-sm);
}

.chat-bubble.assistant-bubble strong {
    color: var(--terracotta) !important;
    font-weight: 600;
}

/* Avatar labels */
.chat-avatar {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.3rem;
    padding: 0 0.2rem;
}

/* Typing indicator */
.typing-indicator {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 0.9rem 1.1rem;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    border-bottom-left-radius: 4px;
    width: fit-content;
    box-shadow: var(--shadow-sm);
}

.typing-dot {
    width: 7px;
    height: 7px;
    background: var(--text-muted);
    border-radius: 50%;
    animation: typing 1.4s infinite;
}

.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }

@keyframes typing {
    0%, 60%, 100% { opacity: 0.2; transform: scale(0.85); }
    30%            { opacity: 1;   transform: scale(1); }
}

/* ============================================================
   FOOD RECOMMENDATION CARDS
   ============================================================ */
.food-cards-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: 1rem;
    margin-top: 1rem;
}

.food-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    overflow: hidden;
    box-shadow: var(--shadow-sm);
    transition: all 0.25s ease;
    position: relative;
}

.food-card:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-lg);
    border-color: var(--gold-lt);
}

.food-card-emoji {
    font-size: 2.8rem;
    text-align: center;
    padding: 1.4rem 1rem 0.6rem;
    line-height: 1;
    display: block;
}

.food-card-body {
    padding: 0.2rem 1.1rem 1.2rem;
}

.food-card-name {
    font-family: var(--font-display) !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    color: var(--brown-dark) !important;
    margin: 0 0 0.3rem !important;
    line-height: 1.3;
}

.food-card-cuisine {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--terracotta);
    margin-bottom: 0.5rem;
}

.food-card-desc {
    font-size: 0.85rem;
    color: var(--text-secondary);
    line-height: 1.55;
    margin-bottom: 0.75rem;
}

.food-card-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
}

.food-tag {
    font-size: 0.72rem;
    padding: 0.2rem 0.55rem;
    background: var(--cream-dark);
    color: var(--brown-mid);
    border-radius: var(--radius-pill);
    font-weight: 500;
}

/* Match score badge */
.food-card-score {
    position: absolute;
    top: 0.75rem;
    right: 0.75rem;
    background: var(--gold);
    color: var(--brown-dark);
    font-size: 0.68rem;
    font-weight: 700;
    padding: 0.15rem 0.5rem;
    border-radius: var(--radius-pill);
    letter-spacing: 0.05em;
}

/* ============================================================
   CHAT INPUT
   ============================================================ */
.stChatInput, [data-testid="stChatInput"] {
    border-radius: var(--radius-pill) !important;
    border: 1.5px solid var(--border) !important;
    background: var(--bg-card) !important;
    box-shadow: var(--shadow-sm) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}

.stChatInput:focus-within, [data-testid="stChatInput"]:focus-within {
    border-color: var(--terracotta) !important;
    box-shadow: 0 0 0 3px rgba(196, 87, 42, 0.12) !important;
}

[data-testid="stChatInput"] textarea {
    font-family: var(--font-body) !important;
    font-size: 0.95rem !important;
    color: var(--text-primary) !important;
}

[data-testid="stChatInput"] button {
    background: var(--terracotta) !important;
    border-radius: var(--radius-pill) !important;
}

/* ============================================================
   STREAMLIT BUTTONS
   ============================================================ */
.stButton > button {
    font-family: var(--font-body) !important;
    font-weight: 500 !important;
    border-radius: var(--radius-pill) !important;
    border: 1.5px solid var(--border) !important;
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.01em !important;
}

.stButton > button:hover {
    background: var(--cream-dark) !important;
    border-color: var(--terracotta) !important;
    color: var(--terracotta) !important;
    transform: translateY(-1px);
    box-shadow: var(--shadow-sm) !important;
}

/* Primary buttons (e.g., Clear Chat) */
.stButton > button[kind="primary"] {
    background: var(--terracotta) !important;
    border-color: var(--terracotta) !important;
    color: white !important;
}

.stButton > button[kind="primary"]:hover {
    background: var(--brown-mid) !important;
    border-color: var(--brown-mid) !important;
    color: white !important;
}

/* ============================================================
   SELECT BOXES & FORM INPUTS
   ============================================================ */
.stSelectbox > div > div {
    border-radius: var(--radius-sm) !important;
    border-color: var(--border) !important;
    font-family: var(--font-body) !important;
}

/* ============================================================
   SECTION HEADINGS
   ============================================================ */
h1, h2, h3 {
    font-family: var(--font-display) !important;
    color: var(--brown-dark) !important;
}

/* ============================================================
   DIVIDERS
   ============================================================ */
hr {
    border-color: var(--border) !important;
    margin: 1.5rem 0 !important;
}

/* ============================================================
   ALERTS & INFO BOXES
   ============================================================ */
.stAlert {
    border-radius: var(--radius-md) !important;
    border-left-color: var(--terracotta) !important;
    background: var(--cream-dark) !important;
    font-family: var(--font-body) !important;
}

/* ============================================================
   METRICS (sidebar stats)
   ============================================================ */
[data-testid="stMetricValue"] {
    font-family: var(--font-display) !important;
    color: var(--gold-lt) !important;
    font-size: 1.5rem !important;
}

[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: rgba(250,246,240,0.6) !important;
}

/* ============================================================
   SPINNER
   ============================================================ */
.stSpinner > div {
    border-top-color: var(--terracotta) !important;
}

/* ============================================================
   WELCOME CARD
   ============================================================ */
.welcome-card {
    background: linear-gradient(135deg, var(--cream-dark) 0%, #FFF8F0 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 2rem;
    text-align: center;
    margin: 1.5rem 0;
    box-shadow: var(--shadow-sm);
}

.welcome-card p {
    font-family: var(--font-display);
    font-style: italic;
    font-size: 1.1rem;
    color: var(--text-secondary);
    line-height: 1.7;
    margin: 0;
}

.welcome-card .welcome-emoji {
    font-size: 3rem;
    display: block;
    margin-bottom: 1rem;
}

/* ============================================================
   SCROLLBAR
   ============================================================ */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--cream); }
::-webkit-scrollbar-thumb {
    background: var(--border);
    border-radius: var(--radius-pill);
}
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

/* ============================================================
   RESPONSIVE
   ============================================================ */
@media (max-width: 640px) {
    .moodbite-logo { font-size: 2.2rem !important; }
    .food-cards-grid { grid-template-columns: 1fr; }
    .chat-message { max-width: 95%; }
    .main .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
}
</style>
"""


def inject_css() -> None:
    """Inject the main CSS into the Streamlit app."""
    import streamlit as st
    st.markdown(MAIN_CSS, unsafe_allow_html=True)