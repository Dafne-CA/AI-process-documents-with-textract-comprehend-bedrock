# utils/ui_components.py
import streamlit as st
import os
from utils.bedrock_agents import invoke_supervisor_agent

# --- Card moderno con colores RPP ---
def card(title, body, buttons=None):
    btn_html = ""
    if buttons:
        for btn in buttons:
            btn_html += f"""
            <button onclick="window.dispatchEvent(new CustomEvent('{btn['key']}'))" 
                    style="margin:3px;padding:6px 12px;border:none;border-radius:8px;
                    background: linear-gradient(135deg, #E31E24 0%, #C71A1F 100%);
                    color:white;font-weight:bold;cursor:pointer;
                    box-shadow: 0 2px 8px rgba(227, 30, 36, 0.3);
                    transition: all 0.3s ease;">
                {btn['label']}
            </button>
            """
    st.markdown(f"""
    <div style='
        background:rgba(255,255,255,0.95); 
        border-radius:16px; 
        padding:1.25rem; 
        box-shadow:0 4px 20px rgba(0,0,0,0.15); 
        margin-bottom:1rem;
        border-left: 5px solid #FFD700;
        transition: all 0.3s ease;'>
        <h4 style='color:#1A1A1A; font-weight:700; margin-bottom:0.75rem;'>{title}</h4>
        <div style='margin:8px 0; color:#4A4A4A;'>{body}</div>
        <div>{btn_html}</div>
    </div>
    """, unsafe_allow_html=True)

# --- Chat flotante con diseño RPP ---
def floating_chat():
    st.markdown("""
    <style>
      .float-btn { 
        position: fixed; 
        right: 2rem; 
        bottom: 2rem; 
        z-index: 9999;
        width: 65px;
        height: 65px;
        background: linear-gradient(135deg, #E31E24 0%, #C71A1F 100%);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 8px 30px rgba(227, 30, 36, 0.5);
        cursor: pointer;
        transition: all 0.3s ease;
        animation: pulse 2s infinite;
      }
      
      .float-btn:hover {
        transform: scale(1.1);
        box-shadow: 0 12px 40px rgba(227, 30, 36, 0.7);
      }
      
      @keyframes pulse {
        0%, 100% { box-shadow: 0 8px 30px rgba(227, 30, 36, 0.5); }
        50% { box-shadow: 0 8px 40px rgba(227, 30, 36, 0.8); }
      }
      
      .chat-box { 
        position: fixed; 
        right: 2rem; 
        bottom: 6rem; 
        z-index: 9999; 
        width: 380px; 
        background: white; 
        border-radius: 16px; 
        box-shadow: 0 8px 40px rgba(0,0,0,0.2); 
        padding: 1rem;
        border-top: 4px solid #E31E24;
      }
      
      .chat-header { 
        font-weight: 700; 
        margin-bottom: 1rem; 
        color: #1A1A1A; 
        font-size: 1.1rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #FFD700;
      }
      
      .chat-input { 
        width: 100%; 
        padding: 0.75rem; 
        border-radius: 8px; 
        border: 2px solid #E5E7EB; 
        margin-bottom: 0.75rem;
        font-size: 0.9rem;
        transition: all 0.3s ease;
      }
      
      .chat-input:focus {
        border-color: #E31E24;
        outline: none;
        box-shadow: 0 0 0 3px rgba(227, 30, 36, 0.1);
      }
      
      .chat-send { 
        background: linear-gradient(135deg, #E31E24 0%, #C71A1F 100%);
        color: white; 
        padding: 0.75rem 1.5rem; 
        border: none; 
        border-radius: 8px; 
        cursor: pointer; 
        font-weight: 700;
        width: 100%;
        box-shadow: 0 2px 8px rgba(227, 30, 36, 0.3);
        transition: all 0.3s ease;
        text-transform: uppercase;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
      }
      
      .chat-send:hover {
        background: linear-gradient(135deg, #C71A1F 0%, #A01519 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(227, 30, 36, 0.4);
      }
    </style>
    """, unsafe_allow_html=True)

    if st.button("💬", key="chat_open", help="Abrir chat con agente IA"):
        with st.container():
            st.markdown("<div class='chat-box'><div class='chat-header'>🤖 Chat Legal AI</div></div>", unsafe_allow_html=True)
            prompt = st.text_input("Escribe tu consulta:", key="chat_prompt", placeholder="Ej: ¿Cuáles son los puntos clave?")
            if st.button("Enviar 🚀", key="chat_send"):
                arn = os.getenv("SUPERVISOR_ALIAS_ARN")
                if not arn:
                    st.warning("⚠️ Supervisor Agent no configurado.")
                else:
                    with st.spinner("🤖 Procesando..."):
                        resp = invoke_supervisor_agent(arn, prompt)
                        st.json(resp)