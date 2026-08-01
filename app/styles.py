"""
styles.py
=========
Custom CSS for the chatbot's dark, minimal, professional look. Kept in
its own module so main.py stays focused on app logic rather than styling.

Design intent: a clean production-assistant feel (generous spacing,
restrained color, plain typography, no emoji/icon clutter) rather than a
"demo project" look.
"""

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    /* ---- Overall canvas ---- */
    .stApp {
        background-color: #0E1117;
    }

    /* ---- Header row ---- */
    .app-header {
        padding: 0.25rem 0 1.25rem 0;
        border-bottom: 1px solid #232733;
        margin-bottom: 1.5rem;
    }
    .app-header h1 {
        font-size: 1.5rem;
        font-weight: 600;
        color: #F2F2F2;
        margin-bottom: 0.15rem;
        letter-spacing: -0.01em;
    }
    .app-header p {
        color: #8B8FA3;
        font-size: 0.92rem;
        margin: 0;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background-color: #10131A;
        border-right: 1px solid #232733;
    }
    section[data-testid="stSidebar"] h3 {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #6B7089;
        margin-top: 1.1rem;
        margin-bottom: 0.4rem;
    }

    /* ---- Buttons ---- */
    .stButton > button {
        border-radius: 8px;
        border: 1px solid #2B3040;
        background-color: #171B26;
        color: #E6E6E6;
        font-weight: 500;
        font-size: 0.88rem;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        border-color: #6C63FF;
        color: #FFFFFF;
        background-color: #1B1F2E;
    }
    .stButton > button[kind="primary"] {
        background-color: #6C63FF;
        border-color: #6C63FF;
        color: #FFFFFF;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #5A52E0;
    }

    /* ---- Chat messages ---- */
    div[data-testid="stChatMessage"] {
        background-color: #131722;
        border: 1px solid #1F2430;
        border-radius: 10px;
        padding: 0.25rem 0.5rem;
        margin-bottom: 0.75rem;
    }

    /* ---- Citation caption ---- */
    .citation-line {
        color: #7C82A3;
        font-size: 0.82rem;
        font-style: normal;
        border-left: 2px solid #6C63FF;
        padding-left: 0.6rem;
        margin-top: 0.4rem;
    }

    .grounding-badge {
        display: inline-block;
        font-size: 0.76rem;
        color: #8B8FA3;
        background-color: #171B26;
        border: 1px solid #2B3040;
        border-radius: 6px;
        padding: 0.15rem 0.55rem;
        margin-top: 0.5rem;
    }

    /* ---- Expander (retrieved chunks panel) ---- */
    details {
        background-color: #10131A;
        border: 1px solid #232733;
        border-radius: 8px;
    }

    /* ---- Empty state ---- */
    .empty-state {
        text-align: center;
        padding: 4rem 1rem;
        color: #6B7089;
    }
    .empty-state h3 {
        color: #C7C9D9;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    /* ---- Misc cleanup ---- */
    #MainMenu, footer {visibility: hidden;}
    .block-container {
        padding-top: 2rem;
        max-width: 900px;
    }
</style>
"""
