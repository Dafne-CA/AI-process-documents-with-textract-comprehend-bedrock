import streamlit as st
import boto3
import os
import time
import json
import base64
import pandas as pd
import uuid
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from utils.bedrock_agents import invoke_agent_legacy
from utils.textract_utils import process_files_with_textract, extract_tables_from_result
from utils.comprehend_utils import clasificar_texto, clasificar_multiple_textos
import re
# ============================================
# CONFIGURACIÓN
# ============================================

load_dotenv()

# Configuración AWS
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BUCKET_NAME = os.getenv("BUCKET_NAME")
COMPREHEND_ENDPOINT_ARN = os.getenv("COMPREHEND_ENDPOINT_ARN")
SUPERVISOR_ALIAS_ID = os.getenv("SUPERVISOR_ALIAS_ID")
SUPERVISOR_AGENT_ID = os.getenv("SUPERVISOR_AGENT_ID")

# Configurar clients AWS
session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

comprehend = session.client('comprehend')

st.set_page_config(
    page_title="RPP Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# SISTEMA DE DISEÑO CORPORATIVO PREMIUM
# ============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    :root {
        --rpp-primary: #D41E2C;
        --rpp-primary-hover: #B01825;
        --rpp-primary-light: #FFE8EA;
        --rpp-dark: #0F172A;
        --rpp-dark-secondary: #1E293B;
        --rpp-gray-50: #F8FAFC;
        --rpp-gray-100: #F1F5F9;
        --rpp-gray-200: #E2E8F0;
        --rpp-gray-300: #CBD5E1;
        --rpp-gray-400: #94A3B8;
        --rpp-gray-600: #475569;
        --rpp-gray-700: #334155;
        --rpp-gray-800: #1E293B;
        --rpp-blue: #3B82F6;
        --rpp-success: #10B981;
        --shadow-xs: 0 1px 2px rgba(0, 0, 0, 0.03);
        --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.04), 0 1px 2px rgba(0, 0, 0, 0.02);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.06), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -2px rgba(0, 0, 0, 0.03);
        --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.08), 0 10px 10px -5px rgba(0, 0, 0, 0.02);
    }
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    
    .main {
        background: var(--rpp-gray-50);
        padding: 0;
    }
    
    [data-testid="stAppViewContainer"] {
        background: var(--rpp-gray-50);
    }
    
    /* ============================================
       HERO SECTION - PREMIUM CORPORATE
       ============================================ */
    .hero-section {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #334155 100%);
        padding: 3.5rem 3rem 3rem 3rem;
        margin: -5rem -5rem 3rem -5rem;
        position: relative;
        overflow: hidden;
        border-bottom: 3px solid var(--rpp-primary);
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            radial-gradient(circle at 20% 50%, rgba(212, 30, 44, 0.08) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, rgba(59, 130, 246, 0.05) 0%, transparent 50%);
        pointer-events: none;
    }
    
    .hero-section::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
    }
    
    .hero-content {
        position: relative;
        z-index: 2;
        max-width: 1400px;
        margin: 0 auto;
    }
    
    .hero-header {
        display: flex;
        align-items: center;
        gap: 2rem;
        margin-bottom: 2rem;
    }
    
    .hero-logo-container {
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 1rem 1.5rem;
        display: flex;
        align-items: center;
    }
    
    .hero-logo {
        height: 48px;
        width: auto;
    }
    
    .hero-text {
        flex: 1;
    }
    
    .hero-title {
        font-family: 'Space Grotesk', 'Inter', sans-serif;
        font-size: 2.75rem;
        font-weight: 700;
        color: #FFFFFF;
        margin: 0 0 0.75rem 0;
        line-height: 1.2;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        font-size: 1.125rem;
        color: rgba(255, 255, 255, 0.7);
        margin: 0;
        font-weight: 400;
        line-height: 1.6;
        max-width: 600px;
    }
    
    .hero-badges {
        display: flex;
        gap: 1rem;
        margin-top: 1.5rem;
        flex-wrap: wrap;
    }
    
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(10px);
        color: rgba(255, 255, 255, 0.9);
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-size: 0.813rem;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.1);
        letter-spacing: 0.03em;
    }
    
    .hero-badge-icon {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    /* ============================================
       STATISTICS CARDS - EXECUTIVE STYLE
       ============================================ */
    .stats-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 1.5rem;
        margin: -1rem 0 3rem 0;
        position: relative;
        z-index: 10;
    }
    
    .stat-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 1.75rem 1.5rem;
        border: 1px solid var(--rpp-gray-200);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: var(--shadow-sm);
        position: relative;
        overflow: hidden;
    }
    
    .stat-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--rpp-primary), var(--rpp-blue));
        transform: scaleX(0);
        transform-origin: left;
        transition: transform 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-lg);
        border-color: var(--rpp-gray-300);
    }
    
    .stat-card:hover::before {
        transform: scaleX(1);
    }
    
    .stat-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1rem;
    }
    
    .stat-icon {
        font-size: 1.5rem;
        opacity: 0.8;
    }
    
    .stat-trend {
        font-size: 0.75rem;
        color: var(--rpp-success);
        font-weight: 600;
        background: rgba(16, 185, 129, 0.1);
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
    }
    
    .stat-number {
        font-size: 2.25rem;
        font-weight: 700;
        color: var(--rpp-dark);
        margin: 0 0 0.25rem 0;
        line-height: 1;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .stat-label {
        font-size: 0.875rem;
        color: var(--rpp-gray-600);
        margin: 0;
        font-weight: 500;
    }
    
    /* ============================================
       SECTION CONTAINERS - PREMIUM CARDS
       ============================================ */
    .section-container {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 2.5rem;
        margin: 2rem 0;
        border: 1px solid var(--rpp-gray-200);
        box-shadow: var(--shadow-sm);
        transition: box-shadow 0.3s ease;
    }
    
    .section-container:hover {
        box-shadow: var(--shadow-md);
    }
    
    .section-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 2rem;
        padding-bottom: 1.25rem;
        border-bottom: 1px solid var(--rpp-gray-200);
    }
    
    .section-icon {
        font-size: 1.5rem;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: var(--rpp-primary-light);
        color: var(--rpp-primary);
        border-radius: 10px;
        font-weight: 600;
    }
    
    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--rpp-dark);
        margin: 0;
        letter-spacing: -0.01em;
    }
    
    .section-subtitle {
        font-size: 0.875rem;
        color: var(--rpp-gray-600);
        margin: -0.25rem 0 0 0;
        font-weight: 400;
    }
    
    /* ============================================
       FILE UPLOAD ZONE - PROFESSIONAL
       ============================================ */
    .upload-zone {
        background: var(--rpp-gray-50);
        border: 2px dashed var(--rpp-gray-300);
        border-radius: 12px;
        padding: 2.5rem 2rem;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
        position: relative;
    }
    
    .upload-zone:hover {
        border-color: var(--rpp-primary);
        background: var(--rpp-primary-light);
    }
    
    .upload-zone-icon {
        font-size: 3rem;
        color: var(--rpp-gray-400);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .upload-zone:hover .upload-zone-icon {
        color: var(--rpp-primary);
        transform: scale(1.1);
    }
    
    .upload-zone-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: var(--rpp-dark);
        margin-bottom: 0.5rem;
    }
    
    .upload-zone-subtitle {
        font-size: 0.875rem;
        color: var(--rpp-gray-600);
        margin: 0;
    }
    
    /* Estilizar el file uploader nativo */
    [data-testid="stFileUploader"] {
        border: none;
        background: transparent;
    }
    
    [data-testid="stFileUploader"] > div {
        padding: 0;
    }
    
    [data-testid="stFileUploadDropzone"] {
        background: var(--rpp-gray-50);
        border: 2px dashed var(--rpp-gray-300);
        border-radius: 12px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: var(--rpp-primary);
        background: var(--rpp-primary-light);
    }
    
    /* ============================================
       FILE PREVIEW GRID - MODERN CARDS
       ============================================ */
    .file-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
        gap: 1.25rem;
        margin: 2rem 0;
    }
    
    .file-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 1.5rem 1.25rem;
        text-align: center;
        border: 1px solid var(--rpp-gray-200);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        position: relative;
        overflow: hidden;
        box-shadow: var(--shadow-xs);
    }
    
    .file-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--rpp-primary);
        transform: scaleX(0);
        transition: transform 0.3s ease;
    }
    
    .file-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-md);
        border-color: var(--rpp-primary);
    }
    
    .file-card:hover::before {
        transform: scaleX(1);
    }
    
    .file-card.selected {
        border-color: var(--rpp-primary);
        background: var(--rpp-primary-light);
        box-shadow: var(--shadow-md);
    }
    
    .file-card.selected::before {
        transform: scaleX(1);
    }
    
    .file-icon {
        font-size: 2.5rem;
        margin-bottom: 0.75rem;
        opacity: 0.8;
        transition: transform 0.3s ease;
    }
    
    .file-card:hover .file-icon {
        transform: scale(1.15);
    }
    
    .file-name {
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--rpp-dark);
        margin: 0.5rem 0 0.25rem 0;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        max-width: 100%;
    }
    
    .file-size {
        font-size: 0.75rem;
        color: var(--rpp-gray-600);
        font-weight: 500;
        margin: 0;
    }
    
    .file-type {
        display: inline-block;
        font-size: 0.688rem;
        color: var(--rpp-primary);
        font-weight: 700;
        margin-top: 0.5rem;
        padding: 0.25rem 0.5rem;
        background: var(--rpp-primary-light);
        border-radius: 4px;
        letter-spacing: 0.05em;
    }
    
    /* ============================================
       BUTTONS - EXECUTIVE STYLE
       ============================================ */
    .stButton > button {
        background: var(--rpp-primary);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.875rem 2rem;
        font-weight: 600;
        font-size: 0.938rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: var(--shadow-sm);
        letter-spacing: 0.01em;
    }
    
    .stButton > button:hover {
        background: var(--rpp-primary-hover);
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* ============================================
       METRICS PANEL - CLEAN DESIGN
       ============================================ */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .metric-card {
        background: var(--rpp-gray-50);
        border-radius: 10px;
        padding: 1.25rem 1rem;
        text-align: center;
        border: 1px solid var(--rpp-gray-200);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        background: #FFFFFF;
        transform: translateY(-2px);
        box-shadow: var(--shadow-sm);
    }
    
    .metric-value {
        font-size: 1.875rem;
        font-weight: 700;
        color: var(--rpp-primary);
        margin: 0.5rem 0;
        line-height: 1;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: var(--rpp-gray-600);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        margin: 0;
    }
    
    /* ============================================
       CLASSIFICATION CARDS - PREMIUM
       ============================================ */
    .classification-card {
        background: #FFFFFF;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid var(--rpp-gray-200);
        transition: all 0.3s ease;
        box-shadow: var(--shadow-xs);
    }
    
    .classification-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-sm);
    }
    
    .classification-icon {
        font-size: 1.75rem;
        margin-bottom: 0.75rem;
    }
    
    .classification-label {
        font-weight: 700;
        font-size: 0.875rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    
    .confidence-bar-container {
        background: var(--rpp-gray-100);
        border-radius: 8px;
        height: 8px;
        overflow: hidden;
        margin-top: 0.5rem;
    }
    
    .confidence-bar {
        height: 100%;
        border-radius: 8px;
        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* ============================================
       CHAT INTERFACE - COPILOT STYLE PRO ✨
       ============================================ */
    .chat-container {
        background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%);
        border-radius: 24px;
        padding: 0;
        overflow: hidden;
        border: 1px solid rgba(59, 130, 246, 0.1);
        box-shadow: 
            0 20px 60px rgba(59, 130, 246, 0.08),
            0 0 0 1px rgba(255, 255, 255, 0.5) inset;
    }
    
    .chat-header {
        background: linear-gradient(135deg, #1E40AF 0%, #3B82F6 50%, #60A5FA 100%);
        padding: 2rem 2.5rem;
        position: relative;
        overflow: hidden;
    }
    
    .chat-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 70%);
        border-radius: 50%;
        animation: float 6s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0) scale(1); }
        50% { transform: translateY(-20px) scale(1.05); }
    }
    
    .chat-header-content {
        position: relative;
        z-index: 2;
        display: flex;
        align-items: center;
        gap: 1.25rem;
    }
    
    .chat-avatar {
        width: 56px;
        height: 56px;
        background: linear-gradient(135deg, #FFFFFF 0%, rgba(255,255,255,0.9) 100%);
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.75rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
        animation: pulse-avatar 3s ease-in-out infinite;
    }
    
    @keyframes pulse-avatar {
        0%, 100% { transform: scale(1); box-shadow: 0 8px 24px rgba(0,0,0,0.15); }
        50% { transform: scale(1.05); box-shadow: 0 12px 32px rgba(59,130,246,0.3); }
    }
    
    .chat-header-text h2 {
        color: #FFFFFF;
        font-size: 1.625rem;
        font-weight: 700;
        margin: 0 0 0.25rem 0;
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: -0.01em;
    }
    
    .chat-header-text p {
        color: rgba(255, 255, 255, 0.85);
        font-size: 0.938rem;
        margin: 0;
        font-weight: 400;
    }
    
    .chat-status {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        padding: 0.5rem 1rem;
        border-radius: 100px;
        font-size: 0.813rem;
        font-weight: 600;
        color: #FFFFFF;
        margin-top: 0.75rem;
    }
    
    .chat-status-dot {
        width: 8px;
        height: 8px;
        background: #10B981;
        border-radius: 50%;
        animation: pulse-dot 2s infinite;
    }
    
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.6; transform: scale(1.2); }
    }
    
    /* ============================================
       SUGERENCIAS - ESTILO CHATGPT/COPILOT 🆕
       ============================================ */
    .suggestions-section {
        padding: 1.5rem 2.5rem;
        background: #FFFFFF;
    }
    
    .suggestions-title {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1rem;
    }
    
    .suggestions-title h3 {
        font-size: 1.125rem;
        font-weight: 700;
        color: var(--rpp-dark);
        margin: 0;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .suggestions-title .sparkle {
        font-size: 1.5rem;
        animation: sparkle 2s ease-in-out infinite;
    }
    
    @keyframes sparkle {
        0%, 100% { transform: rotate(0deg) scale(1); opacity: 1; }
        50% { transform: rotate(180deg) scale(1.1); opacity: 0.8; }
    }
    
    /* Estilos para botones de sugerencias - Cards clickeables 🆕 */
    div[data-testid="column"] .stButton button {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%) !important;
        color: var(--rpp-dark) !important;
        border: 2px solid #E2E8F0 !important;
        border-radius: 12px !important;
        padding: 1rem 1.25rem !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        text-align: left !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
        height: auto !important;
        min-height: 60px !important;
        white-space: normal !important;
        line-height: 1.4 !important;
    }
    
    div[data-testid="column"] .stButton button:hover {
        transform: translateY(-4px) scale(1.02) !important;
        border-color: #3B82F6 !important;
        box-shadow: 0 12px 24px rgba(59, 130, 246, 0.15) !important;
        background: linear-gradient(135deg, #FFFFFF 0%, #EFF6FF 100%) !important;
    }
    
    /* ============================================
       MENSAJES DEL CHAT
       ============================================ */
    .chat-messages {
        padding: 1.5rem 2.5rem;
        background: #FFFFFF;
        min-height: 200px;
        max-height: 450px;
        overflow-y: auto;
    }
    
    .chat-messages::-webkit-scrollbar {
        width: 6px;
    }
    
    .chat-messages::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%);
        border-radius: 100px;
    }
    
    .user-message {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
        animation: slideInRight 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .user-message-avatar {
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, var(--rpp-primary) 0%, #B01825 100%);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        flex-shrink: 0;
        box-shadow: 0 4px 12px rgba(212, 30, 44, 0.2);
    }
    
    .user-message-content {
        flex: 1;
        background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        border: 1px solid var(--rpp-gray-200);
        position: relative;
    }
    
    .user-message-content::before {
        content: '';
        position: absolute;
        left: -8px;
        top: 14px;
        width: 0;
        height: 0;
        border-style: solid;
        border-width: 8px 8px 8px 0;
        border-color: transparent var(--rpp-gray-200) transparent transparent;
    }
    
    .user-message-content::after {
        content: '';
        position: absolute;
        left: -7px;
        top: 15px;
        width: 0;
        height: 0;
        border-style: solid;
        border-width: 7px 7px 7px 0;
        border-color: transparent #F8FAFC transparent transparent;
    }
    
    .assistant-message {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
        animation: slideInLeft 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .assistant-message-avatar {
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        flex-shrink: 0;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.25);
        animation: pulse-avatar 3s ease-in-out infinite;
    }
    
    .assistant-message-content {
        flex: 1;
        background: linear-gradient(135deg, #FFFFFF 0%, #EFF6FF 100%);
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        border: 1px solid rgba(59, 130, 246, 0.2);
        position: relative;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.08);
    }
    
    .assistant-message-content::before {
        content: '';
        position: absolute;
        left: -8px;
        top: 14px;
        width: 0;
        height: 0;
        border-style: solid;
        border-width: 8px 8px 8px 0;
        border-color: transparent rgba(59, 130, 246, 0.2) transparent transparent;
    }
    
    .assistant-message-content::after {
        content: '';
        position: absolute;
        left: -7px;
        top: 15px;
        width: 0;
        height: 0;
        border-style: solid;
        border-width: 7px 7px 7px 0;
        border-color: transparent #FFFFFF transparent transparent;
    }
    
    .message-label {
        font-weight: 700;
        font-size: 0.813rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        letter-spacing: 0.02em;
    }
    
    .user-message-content .message-label {
        color: var(--rpp-primary);
    }
    
    .assistant-message-content .message-label {
        color: #2563EB;
    }
    
    .message-text {
        color: var(--rpp-dark);
        font-size: 0.938rem;
        line-height: 1.7;
        margin: 0;
    }
    
    .chat-input-section {
        padding: 1.5rem 2.5rem;
        background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
        border-top: 1px solid var(--rpp-gray-200);
    }
    
    .chat-input-label {
        font-size: 0.938rem;
        font-weight: 600;
        color: var(--rpp-dark);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* ============================================
       SIDEBAR - EXECUTIVE PANEL
       ============================================ */
    [data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid var(--rpp-gray-200);
        padding-top: 2rem;
    }
    
    .sidebar-section {
        background: var(--rpp-gray-50);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid var(--rpp-gray-200);
    }
    
    .sidebar-title {
        font-weight: 700;
        color: var(--rpp-dark);
        margin-bottom: 1rem;
        font-size: 0.938rem;
        letter-spacing: 0.02em;
    }
    
    .sidebar-content {
        font-size: 0.875rem;
        color: var(--rpp-gray-600);
        background: #FFFFFF;
        padding: 0.75rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid var(--rpp-gray-200);
        font-weight: 500;
    }
    
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.875rem;
        font-weight: 600;
        padding: 0.5rem 0.75rem;
        border-radius: 8px;
    }
    
    .status-connected {
        color: var(--rpp-success);
        background: rgba(16, 185, 129, 0.1);
    }
    
    .status-disconnected {
        color: #EF4444;
        background: rgba(239, 68, 68, 0.1);
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: currentColor;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .capabilities-list {
        font-size: 0.875rem;
        color: var(--rpp-gray-600);
        line-height: 2;
    }
    
    .capability-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.25rem 0;
    }
    
    .capability-icon {
        color: var(--rpp-primary);
        font-size: 1rem;
    }
    
    /* ============================================
       EXPANDER STYLING
       ============================================ */
    .streamlit-expanderHeader {
        background: var(--rpp-gray-50);
        border-radius: 10px;
        border: 1px solid var(--rpp-gray-200);
        font-weight: 600;
        color: var(--rpp-dark);
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: #FFFFFF;
        border-color: var(--rpp-primary);
    }
    
    /* ============================================
       TEXT AREAS & INPUTS
       ============================================ */
    .stTextArea textarea {
        border: 1px solid var(--rpp-gray-300);
        border-radius: 10px;
        font-size: 0.875rem;
        padding: 0.875rem;
        background: var(--rpp-gray-50);
        transition: all 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: var(--rpp-primary);
        background: #FFFFFF;
        box-shadow: 0 0 0 3px rgba(212, 30, 44, 0.1);
    }
    
    /* ============================================
       DATAFRAME STYLING
       ============================================ */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--rpp-gray-200);
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* ============================================
       RESPONSIVE DESIGN
       ============================================ */
    @media (max-width: 1024px) {
        .hero-title { font-size: 2.25rem; }
        .hero-subtitle { font-size: 1rem; }
        .stats-container { grid-template-columns: repeat(2, 1fr); }
    }
    
    @media (max-width: 768px) {
        .hero-section { padding: 2.5rem 1.5rem; }
        .hero-header { flex-direction: column; gap: 1.5rem; }
        .hero-title { font-size: 2rem; }
        .stats-container { grid-template-columns: 1fr; }
        .file-grid { grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); }
        .section-container { padding: 1.5rem; }
    }
    
    /* ============================================
       LOADING & SPINNER
       ============================================ */
    .stSpinner > div {
        border-color: var(--rpp-primary) !important;
    }
    
    /* ============================================
       SCROLLBAR STYLING
       ============================================ */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--rpp-gray-100);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--rpp-gray-300);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--rpp-gray-400);
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# ESTADO DE LA APLICACIÓN
# ============================================

if 'results' not in st.session_state:
    st.session_state['results'] = []
if 'chat_messages' not in st.session_state:
    st.session_state['chat_messages'] = []
if 'chat_visible' not in st.session_state:
    st.session_state['chat_visible'] = False
if 'processing_stats' not in st.session_state:
    st.session_state['processing_stats'] = {'total_docs': 0, 'total_pages': 0, 'total_words': 0}
if 'selected_file_preview' not in st.session_state:
    st.session_state['selected_file_preview'] = None
if 'file_metrics' not in st.session_state:
    st.session_state['file_metrics'] = {}

#NUEVO:Estado para análisis de proveedores
if 'provider_analysis' not in st.session_state:
    st.session_state['provider_analysis'] = None

# ============================================
# FUNCIONES AUXILIARES
# ============================================

def analyze_with_comprehend(text):
    """Analiza texto con AWS Comprehend para obtener metadatos"""
    if not text:
        return {}
    
    try:
        clasificacion = clasificar_texto(text)
        sentiment_response = comprehend.detect_sentiment(Text=text[:5000], LanguageCode='es')
        entities_response = comprehend.detect_entities(Text=text[:5000], LanguageCode='es')
        
        return {
            'clasificacion_documento': clasificacion,
            'sentiment': sentiment_response.get('Sentiment', 'NEUTRAL'),
            'sentiment_scores': sentiment_response.get('SentimentScore', {}),
            'entities': entities_response.get('Entities', [])[:10],
        }
    except Exception as e:
        return {
            'clasificacion_documento': clasificar_texto(text)
        }

def calculate_file_metrics(result):
    """Calcula métricas dinámicas para un archivo procesado"""
    text = result.get('text', '')
    tables_data = extract_tables_from_result(result)
    forms = result.get('forms', {})
    
    word_count = len(text.split())
    table_count = len(tables_data) if tables_data else 0
    form_fields_count = len([k for k, v in forms.items() if k and k.strip() and v and v.strip()])
    
    return {
        'word_count': word_count,
        'table_count': table_count,
        'form_fields_count': form_fields_count,
    }

def display_file_preview_grid(files):
    """Muestra una cuadrícula de archivos moderna"""
    cols_per_row = 6
    files_per_row = min(cols_per_row, len(files))
    
    for i in range(0, len(files), files_per_row):
        row_files = files[i:i + files_per_row]
        cols = st.columns(files_per_row)
        
        for col_idx, file in enumerate(row_files):
            with cols[col_idx]:
                display_file_card(file)

def display_file_card(file):
    """Muestra una tarjeta individual de archivo con diseño premium"""
    file_size = file.size / 1024
    
    if file.type and file.type.startswith('image/'):
        file_icon = "🖼️"
        file_type = "IMAGE"
    elif file.name.lower().endswith('.pdf'):
        file_icon = "📄"
        file_type = "PDF"
    elif file.name.lower().endswith('.eml'):
        file_icon = "📧"
        file_type = "EMAIL"
    else:
        file_icon = "📁"
        file_type = "DOC"
    
    is_selected = st.session_state.get('selected_file_preview') == file
    
    st.markdown(f"""
    <div class="file-card {'selected' if is_selected else ''}">
        <div class="file-icon">{file_icon}</div>
        <div class="file-name" title="{file.name}">{file.name[:20]}{'...' if len(file.name) > 20 else ''}</div>
        <div class="file-size">{file_size:.1f} KB</div>
        <div class="file-type">{file_type}</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Vista Previa", key=f"preview_{file.name}", use_container_width=True):
        st.session_state['selected_file_preview'] = file
        st.rerun()

def display_selected_preview():
    """Muestra la vista previa del archivo seleccionado"""
    selected_file = st.session_state.get('selected_file_preview')
    if selected_file:
        with st.expander(f"👁️ Vista Previa: {selected_file.name}", expanded=True):
            if selected_file.type and selected_file.type.startswith('image/'):
                st.image(selected_file, use_container_width=True)
            elif selected_file.name.lower().endswith('.pdf'):
                st.markdown(f"""
                <div style="text-align: center; padding: 2.5rem; background: var(--rpp-gray-50); border-radius: 12px; border: 1px solid var(--rpp-gray-200);">
                    <div style="font-size: 4rem; margin-bottom: 1rem; opacity: 0.6;">📄</div>
                    <h4 style="color: var(--rpp-dark); margin-bottom: 0.75rem; font-weight: 600;">{selected_file.name}</h4>
                    <p style="color: var(--rpp-gray-600); margin: 0; font-size: 0.875rem;">Documento PDF listo para procesamiento inteligente</p>
                </div>
                """, unsafe_allow_html=True)

def display_metrics_panel(result, file_idx):
    """Muestra el panel de métricas dinámicas para un archivo procesado"""
    metrics = st.session_state['file_metrics'].get(file_idx, {})
    
    st.markdown("**📊 Métricas de Procesamiento**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics.get('word_count', 0):,}</div>
            <div class="metric-label">Palabras</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics.get('table_count', 0)}</div>
            <div class="metric-label">Tablas</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics.get('form_fields_count', 0)}</div>
            <div class="metric-label">Campos</div>
        </div>
        """, unsafe_allow_html=True)

def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None


#####################ultima agregacion:# ============================================
# FUNCIONES MEJORADAS PARA PROMPTS ESTRUCTURADOS
# ============================================

def get_structured_prompt(query, provider_analysis, documents_text):
    """Genera un prompt altamente estructurado para Bedrock"""
    
    # Construir contexto de proveedores de manera estructurada
    providers_context = build_providers_context(provider_analysis)
    products_context = build_products_context(provider_analysis)
    recommendations_context = build_recommendations_context(provider_analysis)
    
    prompt = f"""
# CONTEXTO Y ROL
Eres un **analista senior de compras y proveedores** especializado en optimización de costos. 
Tu objetivo principal es ayudar a tomar decisiones inteligentes sobre proveedores basadas en datos concretos.

# FORMATO DE RESPUESTA OBLIGATORIO
**SIGUE ESTRICTAMENTE este formato en español:**

## 🎯 Análisis Principal
[Resumen ejecutivo de 2-3 líneas con la conclusión más importante]

## 📊 Datos Comparativos
- **Proveedor recomendado:** [Nombre]
- **Ahorro potencial:** [Monto específico]
- **Categoría:** [Categoría específica]
- **Productos analizados:** [Número]

## 💡 Recomendación Específica
[Recomendación accionable y concreta]

## 📈 Detalles Técnicos
[Análisis detallado con datos específicos]

# INFORMACIÓN DE PROVEEDORES DISPONIBLE
{providers_context}

# ANÁLISIS DE PRODUCTOS POR CATEGORÍA
{products_context}

# RECOMENDACIONES DETECTADAS
{recommendations_context}

# DOCUMENTOS PROCESADOS
{documents_text[:3000]}

# CONSULTA DEL USUARIO: "{query}"

# REGLAS ESTRICTAS:
1. **SOLO usar información de los documentos proporcionados**
2. **NO inventar datos o proveedores**
3. **SER específico con montos y porcentajes**
4. **PRIORIZAR ahorros económicos demostrables**
5. **USAR emojis relevantes para mejor legibilidad**
6. **SI no hay datos suficientes, DECIRLO claramente**
7. **EVITAR lenguaje genérico - ser concreto**
8. **INCLUIR números específicos siempre que sea posible**
"""
    return prompt

def build_providers_context(provider_analysis):
    """Construye contexto estructurado de proveedores"""
    if not provider_analysis:
        return "No se detectaron proveedores en los documentos."
    
    context = "## 📋 PROVEEDORES DETECTADOS:\n"
    for provider in provider_analysis.get('providers', []):
        context += f"\n**🏪 {provider['nombre']}**\n"
        context += f"- 📅 Fecha: {provider['fecha']}\n"
        context += f"- 💰 Total documento: {provider['total']}\n"
        context += f"- 📦 Productos: {len(provider['productos'])}\n"
        if provider.get('ruc') and provider['ruc'] != 'No detectado':
            context += f"- 🆔 RUC: {provider['ruc']}\n"
    
    return context

def build_products_context(provider_analysis):
    """Construye análisis estructurado de productos"""
    if not provider_analysis or not provider_analysis.get('analisis_categorias'):
        return "No hay análisis de productos disponible."
    
    context = "## 📊 ANÁLISIS POR CATEGORÍAS:\n"
    for categoria, datos in provider_analysis['analisis_categorias'].items():
        context += f"\n**📦 {categoria.upper()}**\n"
        context += f"- Precio promedio: S/. {datos.get('precio_promedio', 0):.2f}\n"
        context += f"- Rango: S/. {datos.get('precio_min', 0):.2f} - S/. {datos.get('precio_max', 0):.2f}\n"
        context += f"- Proveedores: {', '.join(list(datos.get('proveedores', []))[:3])}\n"
        context += f"- Total productos: {datos.get('total_productos', 0)}\n"
    
    return context

def build_recommendations_context(provider_analysis):
    """Construye recomendaciones estructuradas"""
    if not provider_analysis or not provider_analysis.get('recomendaciones'):
        return "No hay recomendaciones disponibles."
    
    recs = provider_analysis['recomendaciones']
    context = "## 💡 OPORTUNIDADES IDENTIFICADAS:\n"
    
    if recs.get('mejores_proveedores'):
        context += "\n**🏆 MEJORES PROVEEDORES POR CATEGORÍA:**\n"
        for categoria, mejor in recs['mejores_proveedores'].items():
            context += f"- {categoria}: {mejor['proveedor']} (S/. {mejor.get('precio', 0):.2f})\n"
    
    if recs.get('ahorros_potenciales'):
        context += "\n**💰 AHORROS POTENCIALES:**\n"
        for ahorro in recs['ahorros_potenciales']:
            context += f"- {ahorro['categoria']}: S/. {ahorro.get('ahorro_estimado', 0):.2f} con {ahorro['proveedor_recomendado']}\n"
    
    return context

# ============================================
# FUNCIONES DE ANÁLISIS MEJORADAS CON TEXTRACT
# ============================================

def extract_products_from_tables(tables_data, provider_name):
    """Extrae productos de las tablas detectadas por Textract - MEJORADO"""
    products = []
    
    for table_info in tables_data:
        df = table_info['dataframe']
        
        # Buscar columnas que puedan contener productos y precios
        product_columns = []
        price_columns = []
        quantity_columns = []
        
        for col in df.columns:
            col_str = str(col).lower()
            if any(word in col_str for word in ['producto', 'descripción', 'item', 'concepto', 'servicio']):
                product_columns.append(col)
            elif any(word in col_str for word in ['precio', 'importe', 'valor', 'costo', 'unitario']):
                price_columns.append(col)
            elif any(word in col_str for word in ['cantidad', 'qty', 'unidades']):
                quantity_columns.append(col)
        
        # Si no encontramos columnas específicas, usar heurísticas
        if not product_columns:
            # Buscar columnas con texto que parezcan productos
            for col in df.columns:
                sample_values = df[col].dropna().head(3).astype(str)
                if any(len(str(val)) > 10 and any(char.isalpha() for char in str(val)) for val in sample_values):
                    product_columns.append(col)
        
        if not price_columns:
            # Buscar columnas con valores numéricos
            for col in df.columns:
                try:
                    numeric_values = pd.to_numeric(df[col].dropna(), errors='coerce')
                    if numeric_values.notna().sum() > 0:
                        price_columns.append(col)
                except:
                    pass
        
        # Extraer productos
        for product_col in product_columns[:1]:  # Usar solo la primera columna de productos
            for idx, row in df.iterrows():
                product_name = str(row[product_col]).strip()
                if (product_name and 
                    product_name not in ['', 'nan', 'None'] and 
                    len(product_name) > 2 and
                    not any(word in product_name.lower() for word in ['total', 'subtotal', 'igv', 'impuesto'])):
                    
                    # Buscar precio
                    price = None
                    for price_col in price_columns:
                        try:
                            price_val = str(row[price_col]).replace(',', '').replace('S/', '').replace('$', '').strip()
                            if price_val and price_val not in ['', 'nan', 'None']:
                                price = float(price_val)
                                break
                        except:
                            continue
                    
                    # Buscar cantidad
                    quantity = None
                    for qty_col in quantity_columns:
                        qty_val = str(row[qty_col]).strip()
                        if qty_val and qty_val not in ['', 'nan', 'None']:
                            quantity = qty_val
                            break
                    
                    products.append({
                        'nombre': product_name,
                        'precio': price,
                        'cantidad': quantity,
                        'categoria': categorize_product(product_name),
                        'proveedor': provider_name,
                        'fuente': 'tabla'
                    })
    
    return products

def extract_products_from_text(text, provider_name):
    """Extrae productos del texto usando patrones mejorados"""
    products = []
    
    # Patrones para líneas que parecen productos con precios
    patterns = [
        r'([A-Za-z\s\-\&]+)\s+(\d+)[\s,]*(\d+\.\d{2})',  # Producto cantidad precio
        r'([A-Za-z\s\-\&]+)\s+S\/\.\s*(\d+\.\d{2})',     # Producto S/. precio
        r'([A-Za-z\s\-\&]+)\s+\$?\s*(\d+[.,]\d{2})',     # Producto $ precio
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            product_name = match.group(1).strip()
            if len(product_name) > 3:  # Filtrar nombres muy cortos
                price = None
                quantity = None
                
                if len(match.groups()) >= 3:
                    try:
                        price = float(match.group(3).replace(',', '.'))
                        quantity = match.group(2)
                    except:
                        pass
                elif len(match.groups()) >= 2:
                    try:
                        price = float(match.group(2).replace(',', '.'))
                    except:
                        pass
                
                products.append({
                    'nombre': product_name,
                    'precio': price,
                    'cantidad': quantity,
                    'categoria': categorize_product(product_name),
                    'proveedor': provider_name,
                    'fuente': 'texto'
                })
    
    return products

def categorize_product(product_name):
    """Categoriza productos automáticamente"""
    product_name_lower = product_name.lower()
    
    categorias = {
        'gaseosas': ['coca', 'pepsi', 'sprite', 'fanta', 'inca', 'cola', 'gaseosa', 'refresco'],
        'aguas': ['agua', 'cielo', 'cristal', 'mineral', 'aqua'],
        'cervezas': ['pilsen', 'cristal', 'cusqueña', 'heineken', 'corona', 'cerveza', 'lager'],
        'jugos': ['jugo', 'néctar', 'refresco', 'pulp', 'zumo'],
        'lácteos': ['leche', 'yogur', 'queso', 'mantequilla', 'lácteo', 'crema'],
        'carnes': ['pollo', 'carne', 'pescado', 'res', 'cerdo', 'vacuno', 'filete'],
        'granos': ['arroz', 'fideo', 'harina', 'maíz', 'trigo', 'avena', 'quinua'],
        'básicos': ['aceite', 'azúcar', 'sal', 'pan', 'huevo', 'aceituna'],
        'frutas_verduras': ['fruta', 'verdura', 'legumbre', 'vegetal', 'tomate', 'cebolla'],
        'limpieza': ['jabón', 'detergente', 'limpiador', 'cloro', 'lavavajilla'],
        'electrónicos': ['tv', 'televisor', 'celular', 'tablet', 'laptop', 'computadora']
    }
    
    for categoria, palabras in categorias.items():
        if any(palabra in product_name_lower for palabra in palabras):
            return categoria
    
    return 'otros'

def extract_provider_info_advanced(text, filename, tables_data=None, forms_data=None):
    """Extrae información del proveedor usando Textract + Comprehend - MEJORADA"""
    provider_info = {
        'nombre': 'Desconocido',
        'fecha': 'No detectada',
        'total': 'No detectado',
        'productos': [],
        'filename': filename,
        'ruc': 'No detectado',
        'direccion': 'No detectada',
        'tipo_documento': 'desconocido'
    }
    
    # Usar Comprehend para análisis de entidades
    try:
        entities = comprehend.detect_entities(Text=text[:5000], LanguageCode='es')
        for entity in entities['Entities']:
            if entity['Type'] == 'ORGANIZATION' and provider_info['nombre'] == 'Desconocido':
                provider_info['nombre'] = entity['Text']
            elif entity['Type'] == 'DATE' and provider_info['fecha'] == 'No detectada':
                provider_info['fecha'] = entity['Text']
            elif entity['Type'] == 'COMMERCIAL_ITEM' and 'factura' in entity['Text'].lower():
                provider_info['tipo_documento'] = 'factura'
    except:
        pass
    
    # Patrones mejorados para información del proveedor
    provider_patterns = [
        r'PROVEEDOR[:\s]+([^\n]+)',
        r'EMISOR[:\s]+([^\n]+)',
        r'RAZ[ÓO]N SOCIAL[:\s]+([^\n]+)',
        r'EMPRESA[:\s]+([^\n]+)',
        r'VENDEDOR[:\s]+([^\n]+)'
    ]
    
    for pattern in provider_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            provider_name = match.group(1) if match.lastindex >= 1 else None
            if provider_name is not None and len(provider_name.strip()) > 3:
                provider_info['nombre'] = provider_name.strip()
                break
    
    # Buscar RUC
    ruc_patterns = [
        r'RUC[:\s]*([0-9]{11})',
        r'R\.U\.C\.?[:\s]*([0-9]{11})',
    ]
    
    for pattern in ruc_patterns:
        match = re.search(pattern, text)
        if match and match.group(1) is not None:
            provider_info['ruc'] = match.group(1)
            break
    
    # Buscar fecha
    date_patterns = [
        r'FECHA[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        r'FECHA DE EMISI[ÓO]N[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match and match.group(1) is not None:
            provider_info['fecha'] = match.group(1)
            break
    
    # Buscar total mejorado
    total_patterns = [
        r'TOTAL[:\s]*[\$S/\.\s]*([\d,]+(?:\.\d{2})?)',
        r'IMPORTE TOTAL[:\s]*[\$S/\.\s]*([\d,]+(?:\.\d{2})?)',
        r'MONTO TOTAL[:\s]*[\$S/\.\s]*([\d,]+(?:\.\d{2})?)',
    ]
    
    for pattern in total_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match and match.group(1) is not None:
            try:
                provider_info['total'] = float(match.group(1).replace(',', ''))
            except:
                provider_info['total'] = match.group(1)
            break
    
    # Extraer productos de múltiples fuentes
    all_products = []
    
    # 1. De tablas (más confiable)
    if tables_data:
        table_products = extract_products_from_tables(tables_data, provider_info['nombre'])
        all_products.extend(table_products)
    
    # 2. Del texto (como respaldo)
    text_products = extract_products_from_text(text, provider_info['nombre'])
    all_products.extend(text_products)
    
    # Eliminar duplicados
    unique_products = []
    seen_products = set()
    for product in all_products:
        product_key = f"{product['nombre']}_{product['precio']}"
        if product_key not in seen_products:
            seen_products.add(product_key)
            unique_products.append(product)
    
    provider_info['productos'] = unique_products
    
    return provider_info

def analyze_providers_comparison_advanced(results):
    """Analiza y compara múltiples proveedores usando Textract + Bedrock"""
    providers = []
    all_products = []
    
    for result in results:
        text = result.get('text', '')
        if text and text.strip():
            tables_data = extract_tables_from_result(result)
            provider_info = extract_provider_info_advanced(
                text, 
                result['filename'], 
                tables_data,
                result.get('forms', {})
            )
            
            # Solo agregar proveedores con información válida
            if (provider_info['nombre'] != 'Desconocido' or 
                provider_info['productos'] or
                provider_info['total'] != 'No detectado'):
                providers.append(provider_info)
                
                # Recolectar todos los productos para análisis
                for producto in provider_info['productos']:
                    all_products.append(producto)
    
    # Análisis por categorías
    categorias_analisis = analyze_categories(all_products)
    
    # Generar recomendaciones
    recomendaciones = generate_ai_recommendations(providers, all_products)
    
    # Análisis de precios comparativos
    price_analysis = analyze_prices_comparison(all_products)
    
    return {
        'total_providers': len(providers),
        'providers': providers,
        'productos_totales': all_products,
        'analisis_categorias': categorias_analisis,
        'recomendaciones': recomendaciones,
        'analisis_precios': price_analysis
    }

def analyze_categories(products):
    """Analiza productos por categorías"""
    categorias = {}
    
    for producto in products:
        categoria = producto['categoria']
        if categoria not in categorias:
            categorias[categoria] = {
                'productos': [],
                'proveedores': set(),
                'precio_promedio': 0,
                'precio_min': float('inf'),
                'precio_max': 0,
                'total_productos': 0
            }
        
        categorias[categoria]['productos'].append(producto)
        categorias[categoria]['proveedores'].add(producto['proveedor'])
        categorias[categoria]['total_productos'] += 1
        
        # Estadísticas de precios
        if producto['precio']:
            precio = producto['precio']
            categorias[categoria]['precio_promedio'] += precio
            categorias[categoria]['precio_min'] = min(categorias[categoria]['precio_min'], precio)
            categorias[categoria]['precio_max'] = max(categorias[categoria]['precio_max'], precio)
    
    # Calcular promedios
    for categoria in categorias:
        productos_con_precio = [p for p in categorias[categoria]['productos'] if p['precio']]
        if productos_con_precio:
            categorias[categoria]['precio_promedio'] /= len(productos_con_precio)
        else:
            categorias[categoria]['precio_promedio'] = 0
            categorias[categoria]['precio_min'] = 0
            categorias[categoria]['precio_max'] = 0
    
    return categorias

def analyze_prices_comparison(products):
    """Analiza comparación de precios entre proveedores"""
    price_analysis = {}
    
    for producto in products:
        if producto['precio']:
            nombre_producto = producto['nombre']
            if nombre_producto not in price_analysis:
                price_analysis[nombre_producto] = {
                    'proveedores': [],
                    'precios': [],
                    'precio_min': float('inf'),
                    'precio_max': 0,
                    'proveedor_mas_barato': None
                }
            
            price_analysis[nombre_producto]['proveedores'].append(producto['proveedor'])
            price_analysis[nombre_producto]['precios'].append(producto['precio'])
            
            # Actualizar min/max
            if producto['precio'] < price_analysis[nombre_producto]['precio_min']:
                price_analysis[nombre_producto]['precio_min'] = producto['precio']
                price_analysis[nombre_producto]['proveedor_mas_barato'] = producto['proveedor']
            
            if producto['precio'] > price_analysis[nombre_producto]['precio_max']:
                price_analysis[nombre_producto]['precio_max'] = producto['precio']
    
    return price_analysis

def generate_ai_recommendations(providers, products):
    """Genera recomendaciones inteligentes"""
    if not providers or len(providers) < 2:
        return {
            'mejores_proveedores': {},
            'ahorros_potenciales': [],
            'alertas': ['Se necesitan al menos 2 proveedores para comparación']
        }
    
    recomendaciones = {
        'mejores_proveedores': {},
        'ahorros_potenciales': [],
        'alertas': []
    }
    
    # Análisis por categoría
    categorias = {}
    for producto in products:
        if producto['categoria'] not in categorias:
            categorias[producto['categoria']] = []
        categorias[producto['categoria']].append(producto)
    
    # Encontrar mejores precios por categoría
    for categoria, productos_cat in categorias.items():
        productos_con_precio = [p for p in productos_cat if p['precio']]
        if productos_con_precio and len(set(p['proveedor'] for p in productos_con_precio)) >= 2:
            mejor_precio = min(productos_con_precio, key=lambda x: x['precio'])
            
            recomendaciones['mejores_proveedores'][categoria] = {
                'proveedor': mejor_precio['proveedor'],
                'producto': mejor_precio['nombre'],
                'precio': mejor_precio['precio'],
                'categoria': categoria
            }
            
            # Calcular ahorro potencial
            precios = [p['precio'] for p in productos_con_precio if p['precio']]
            if len(precios) > 1:
                precio_promedio = sum(precios) / len(precios)
                ahorro_potencial = precio_promedio - mejor_precio['precio']
                if ahorro_potencial > 0:
                    recomendaciones['ahorros_potenciales'].append({
                        'categoria': categoria,
                        'proveedor_recomendado': mejor_precio['proveedor'],
                        'ahorro_estimado': round(ahorro_potencial, 2),
                        'producto_ejemplo': mejor_precio['nombre']
                    })
    
    return recomendaciones



#####################

#esto estoy agregando al chatbot para sugerir respuestas basadas en el contexto del documento

def extract_provider_info(text, filename):
    """Extrae información del proveedor del texto de facturas - CORREGIDA"""
    provider_info = {
        'nombre': 'Desconocido',
        'fecha': 'No detectada',
        'total': 'No detectado',
        'productos': [],
        'filename': filename
    }
    
    
    # Buscar nombre del proveedor (patrones comunes)
    provider_patterns = [
        r'PROVEEDOR[:\s]+([^\n]+)',
        r'EMISOR[:\s]+([^\n]+)',
        r'RAZÓN SOCIAL[:\s]+([^\n]+)',
        r'CLIENTE[:\s]+([^\n]+)',
        r'SEÑOR(ES)?[:\s]+([^\n]+)'
    ]
    
    for pattern in provider_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # CORRECCIÓN: Verificar que group(1) no sea None antes de usar strip()
            provider_name = match.group(1)
            if provider_name is not None:
                provider_info['nombre'] = provider_name.strip()
            break
    
    # Buscar fecha
    date_patterns = [
        r'FECHA[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match and match.group(1) is not None:
            provider_info['fecha'] = match.group(1)
            break
    
    # Buscar total
    total_patterns = [
        r'TOTAL[:\s]*\$?\s*([\d,]+(?:\.\d{2})?)',
        r'IMPORTE TOTAL[:\s]*\$?\s*([\d,]+(?:\.\d{2})?)',
        r'MONTO TOTAL[:\s]*\$?\s*([\d,]+(?:\.\d{2})?)'
    ]
    
    for pattern in total_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match and match.group(1) is not None:
            try:
                provider_info['total'] = float(match.group(1).replace(',', ''))
            except:
                provider_info['total'] = match.group(1)
            break
    
    return provider_info

def analyze_providers_comparison(results):
    """Analiza y compara múltiples proveedores - MEJORADA"""
    providers = []
    
    for result in results:
        text = result.get('text', '')
        if text and text.strip():  # Verificar que el texto no esté vacío
            provider_info = extract_provider_info(text, result['filename'])
            # Solo agregar proveedores con información válida
            if provider_info['nombre'] != 'Desconocido' or provider_info['fecha'] != 'No detectada':
                providers.append(provider_info)
    
    return {
        'total_providers': len(providers),
        'providers': providers
    }
def get_chat_suggestions(results):
    """Genera sugerencias contextuales inteligentes basadas en tipos de documentos"""
    suggestions = []
    
    # Analizar tipos de documentos
    doc_types = []
    for result in results:
        if result.get('comprehend_analysis'):
            clasificacion = result['comprehend_analysis'].get('clasificacion_documento', {})
            doc_type = clasificacion.get('clase', 'desconocido')
            doc_types.append(doc_type)
    
    # Contar tipos específicos
    factura_count = doc_types.count('factura')
    contrato_count = doc_types.count('contrato') 
    legal_count = sum(1 for doc_type in doc_types if doc_type in ['demanda', 'carta_notarial'])
    multiple_docs = len(results) >= 2
    
    # SUGERENCIAS ESPECÍFICAS POR TIPO DE DOCUMENTO
    
    # Para facturas
    if factura_count >= 1:
        suggestions.extend([
            "📊 Comparar proveedores y analizar costos",
            "💰 ¿Cuál proveedor es más conveniente?",
            "📈 Mostrar tendencias de compra",
            "🧾 Resumir montos y fechas de facturas"
        ])
    
    # Para contratos/documentos legales
    if contrato_count >= 1 or legal_count >= 1:
        suggestions.extend([
            "⚖️ Analizar cláusulas contractuales",
            "⚠️ Identificar posibles penalidades",
            "📑 Revisar obligaciones legales", 
            "📅 Verificar fechas y plazos importantes"
        ])
    
    # Para múltiples documentos
    if multiple_docs:
        suggestions.extend([
            "📋 Comparar contenido entre documentos",
            "🔍 Encontrar información común"
        ])
    
    # SUGERENCIAS GENERALES (siempre disponibles)
    general_suggestions = [
        "📋 Resumir el contenido principal",
        "🔍 Buscar información específica",
        "📊 Extraer datos importantes"
    ]
    
    # Combinar y eliminar duplicados
    all_suggestions = suggestions + general_suggestions
    unique_suggestions = list(dict.fromkeys(all_suggestions))
    
    return unique_suggestions[:6]  # Máximo 6 sugerencias

# ============================================
# HERO SECTION - DISEÑO PREMIUM CORPORATIVO
# ============================================

logo_base64 = get_base64_image("assets/rpp.png")

st.markdown(f"""
<div class="hero-section">
    <div class="hero-content">
        <div class="hero-header">
            <div class="hero-logo-container">
                {"<img src='data:image/png;base64,{}' class='hero-logo' alt='RPP'>".format(logo_base64) if logo_base64 else "<div style='color: white; font-weight: 800; font-size: 2rem;'>RPP</div>"}
            </div>
            <div class="hero-text">
                <h1 class="hero-title" style="color: #FFFFFF">PLATAFORMA DE INTELIGENCIA</h1>
                <p class="hero-subtitle">Solución empresarial de procesamiento inteligente de documentos mediante IA generativa y análisis avanzado</p>
            </div>
        </div>
        <div class="hero-badges">
            <div class="hero-badge">
                <span class="hero-badge-icon">⚡</span>
                <span>AWS AI Services</span>
            </div>
            <div class="hero-badge">
                <span class="hero-badge-icon">🔒</span>
                <span>Enterprise Grade</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================
# PANEL DE ESTADÍSTICAS - EXECUTIVE DASHBOARD
# ============================================

st.markdown(f"""
<div class="stats-container">
    <div class="stat-card">
        <div class="stat-header">
            <span class="stat-icon">📄</span>
            <span class="stat-trend">↑ Live</span>
        </div>
        <p class="stat-number">{st.session_state['processing_stats']['total_docs']}</p>
        <p class="stat-label">Documentos Procesados</p>
    </div>
    <div class="stat-card">
        <div class="stat-header">
            <span class="stat-icon">📑</span>
            <span class="stat-trend">↑ Live</span>
        </div>
        <p class="stat-number">{st.session_state['processing_stats']['total_pages']}</p>
        <p class="stat-label">Páginas Analizadas</p>
    </div>
    <div class="stat-card">
        <div class="stat-header">
            <span class="stat-icon">✍️</span>
            <span class="stat-trend">↑ Live</span>
        </div>
        <p class="stat-number">{st.session_state['processing_stats']['total_words']:,}</p>
        <p class="stat-label">Palabras Extraídas</p>
    </div>
    <div class="stat-card">
        <div class="stat-header">
            <span class="stat-icon">🤖</span>
        </div>
        <p class="stat-number">{'ACTIVE' if SUPERVISOR_AGENT_ID else 'STANDBY'}</p>
        <p class="stat-label">AI Agent Status</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR - PANEL DE CONFIGURACIÓN EJECUTIVO
# ============================================

with st.sidebar:
    st.markdown("### ⚙️ Configuración del Sistema")
    
    st.markdown(f"""
    <div class="sidebar-section">
        <div class="sidebar-title">🗄️ Amazon S3 Storage</div>
        <div class="sidebar-content">
            {BUCKET_NAME or 'No configurado'}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if SUPERVISOR_AGENT_ID:
        st.markdown(f"""
        <div class="sidebar-section">
            <div class="sidebar-title">🤖 AI Agent</div>
            <div class="status-indicator status-connected">
                <span class="status-dot"></span>
                <span>Conectado</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="sidebar-section">
            <div class="sidebar-title">🤖 AI Agent</div>
            <div class="status-indicator status-disconnected">
                <span class="status-dot"></span>
                <span>Desconectado</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-title">🚀 Capacidades del Sistema</div>
        <div class="capabilities-list">
            <div class="capability-item">
                <span class="capability-icon">✓</span>
                <span>Extracción OCR Avanzada</span>
            </div>
            <div class="capability-item">
                <span class="capability-icon">✓</span>
                <span>Clasificación Inteligente</span>
            </div>
            <div class="capability-item">
                <span class="capability-icon">✓</span>
                <span>Análisis de Sentimiento</span>
            </div>
            <div class="capability-item">
                <span class="capability-icon">✓</span>
                <span>Extracción de Tablas</span>
            </div>
            <div class="capability-item">
                <span class="capability-icon">✓</span>
                <span>Chat Conversacional IA</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div style="text-align: center; padding: 1rem; color: var(--rpp-gray-600); font-size: 0.75rem;">
        <p style="margin: 0;">RPP Intelligence Platform v2.0</p>
        <p style="margin: 0.25rem 0 0 0;">Powered by AWS AI</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# SECCIÓN DE CARGA DE DOCUMENTOS - PREMIUM
# ============================================

st.markdown("""
<div class="section-container">
    <div class="section-header">
        <span class="section-icon">📤</span>
        <div>
            <h2 class="section-title">Carga de Documentos</h2>
            <p class="section-subtitle">Soporta PDF, imágenes y archivos de correo electrónico</p>
        </div>
    </div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Arrastra o selecciona archivos para procesar",
    accept_multiple_files=True,
    type=["pdf", "png", "jpg", "jpeg", "eml"],
    help="Formatos soportados: PDF, PNG, JPG, JPEG, EML | Tamaño máximo: 10MB por archivo"
)

st.markdown("</div>", unsafe_allow_html=True)
# ============================================
# VISTA PREVIA DE ARCHIVOS - GRID MODERNO
# ============================================

if uploaded_files:
    st.markdown("""
    <div class="section-container">
        <div class="section-header">
            <span class="section-icon">👁️</span>
            <div>
                <h2 class="section-title">Archivos Cargados</h2>
                <p class="section-subtitle">Previsualización y gestión de documentos</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    display_file_preview_grid(uploaded_files)
    display_selected_preview()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🚀 Iniciar Procesamiento Inteligente", use_container_width=True, type="primary"):
        with st.spinner("🔍 Procesando documentos con IA..."):
            results = process_files_with_textract(uploaded_files)
            
            for idx, result in enumerate(results):
                text_content = result.get('text', '')
                if text_content:
                    comprehend_analysis = analyze_with_comprehend(text_content)
                    result['comprehend_analysis'] = comprehend_analysis
                
                metrics = calculate_file_metrics(result)
                st.session_state['file_metrics'][idx] = metrics
            
            st.session_state['results'] = results
            st.session_state['chat_visible'] = True
            
            # 🆕 ANÁLISIS AVANZADO: Usar el nuevo análisis mejorado
            if len(results) >= 1:  # Ahora funciona incluso con un solo documento
                provider_analysis = analyze_providers_comparison_advanced(results)
                st.session_state['provider_analysis'] = provider_analysis
                
                # Mostrar análisis completo de proveedores
                with st.expander("📊 Análisis Avanzado de Proveedores", expanded=True):
                    st.markdown(f"**Se detectaron {provider_analysis['total_providers']} proveedores:**")
                    
                    for provider in provider_analysis['providers']:
                        col1, col2, col3 = st.columns([3, 2, 2])
                        with col1:
                            st.write(f"**{provider['nombre']}**")
                            if provider['ruc'] != 'No detectado':
                                st.write(f"RUC: {provider['ruc']}")
                        with col2:
                            st.write(f"Fecha: {provider['fecha']}")
                            st.write(f"Productos: {len(provider['productos'])}")
                        with col3:
                            st.write(f"Total: {provider['total']}")
            
            st.session_state['processing_stats']['total_docs'] = len(results)
            st.session_state['processing_stats']['total_pages'] = sum(r.get('pages', 1) for r in results)
            st.session_state['processing_stats']['total_words'] = sum(len((r.get('text', '') or '').split()) for r in results)
            
            st.success("✓ Procesamiento completado exitosamente")
            st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================
# RESULTADOS DEL ANÁLISIS - DISEÑO PREMIUM
# ============================================

if st.session_state.get('results'):
    st.markdown("""
    <div class="section-container">
        <div class="section-header">
            <span class="section-icon">📊</span>
            <div>
                <h2 class="section-title">Resultados del Análisis</h2>
                <p class="section-subtitle">Extracción, clasificación y análisis de contenido</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    for idx, result in enumerate(st.session_state['results']):
        with st.expander(f"📄 {result['filename']}", expanded=(idx == 0)):
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown("**📝 Contenido Extraído**")
                text_content = result.get('text', 'No se pudo extraer texto')
                st.text_area(
                    "", 
                    value=text_content,
                    height=250, 
                    key=f"text_{idx}",
                    label_visibility="collapsed"
                )
                
                tables_data = extract_tables_from_result(result)
                if tables_data:
                    st.markdown("---")
                    st.markdown("**📋 Tablas Detectadas**")
                    for table_info in tables_data:
                        st.markdown(f"**Tabla {table_info['index']}**")
                        st.dataframe(table_info['dataframe'], use_container_width=True)
                
                if result.get('forms'):
                    st.markdown("---")
                    st.markdown("**📝 Campos de Formulario**")
                    forms_data = []
                    for key, value in result['forms'].items():
                        if key and key.strip() and value and value.strip():
                            forms_data.append({"Campo": key, "Valor": value})
                    
                    if forms_data:
                        st.dataframe(forms_data, use_container_width=True)
            
            with col2:
                display_metrics_panel(result, idx)
                
                if result.get('comprehend_analysis'):
                    st.markdown("---")
                    st.markdown("**🔍 Análisis de IA**")
                    
                    analysis = result['comprehend_analysis']
                    sentiment = analysis.get('sentiment', 'NEUTRAL')
                    
                    sentiment_colors = {
                        'POSITIVE': '#10B981',
                        'NEGATIVE': '#EF4444', 
                        'NEUTRAL': '#6B7280',
                        'MIXED': '#F59E0B'
                    }
                    
                    sentiment_color = sentiment_colors.get(sentiment, '#6B7280')
                    
                    st.markdown(f"""
                    <div style="background: {sentiment_color}15; border: 1px solid {sentiment_color}30; border-radius: 8px; padding: 0.75rem; text-align: center;">
                        <div style="color: {sentiment_color}; font-weight: 700; font-size: 0.875rem;">🎭 {sentiment}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if analysis.get('clasificacion_documento'):
                        clasificacion = analysis['clasificacion_documento']
                        
                        st.markdown("---")
                        st.markdown("**📄 Clasificación**")
                        
                        clase_config = {
                            'contrato': {'emoji': '📑', 'color': '#3B82F6'},
                            'factura': {'emoji': '🧾', 'color': '#10B981'},
                            'boleta': {'emoji': '🎫', 'color': '#F59E0B'}, 
                            'demanda': {'emoji': '⚖️', 'color': '#EF4444'},
                            'estado_cuenta': {'emoji': '💳', 'color': '#8B5CF6'},
                            'recibo': {'emoji': '🧾', 'color': '#06B6D4'},
                            'carta_notarial': {'emoji': '✉️', 'color': '#F97316'},
                            'desconocido': {'emoji': '❓', 'color': '#6B7280'},
                            'error_comprehend': {'emoji': '⚠️', 'color': '#DC2626'}
                        }
                        
                        clase = clasificacion.get('clase', 'desconocido')
                        confianza = clasificacion.get('confianza', 0)
                        config = clase_config.get(clase, {'emoji': '📄', 'color': '#6B7280'})
                        
                        col_c1, col_c2 = st.columns(2)
                        
                        with col_c1:
                            st.markdown(f"""
                            <div class="classification-card" style="border-color: {config['color']}30; background: {config['color']}08;">
                                <div class="classification-icon">{config['emoji']}</div>
                                <div class="classification-label" style="color: {config['color']};">{clase.upper()}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_c2:
                            confianza_width = int(confianza * 100)
                            color_confianza = (
                                '#10B981' if confianza >= 0.7 else
                                '#F59E0B' if confianza >= 0.4 else
                                '#EF4444'
                            )
                            
                            st.markdown(f"""
                            <div class="classification-card">
                                <div style="font-weight: 700; color: var(--rpp-dark); font-size: 1.25rem; margin-bottom: 0.5rem;">{confianza:.0%}</div>
                                <div class="confidence-bar-container">
                                    <div class="confidence-bar" style="background: {color_confianza}; width: {confianza_width}%;"></div>
                                </div>
                                <div style="font-size: 0.7rem; color: var(--rpp-gray-600); margin-top: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">Confianza</div>
                            </div>
                            """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# # ============================================
# ASISTENTE IA - INTERFAZ DE CHAT MODERNA CON SUGERENCIAS
# ============================================

if st.session_state.get('results') and not st.session_state.get('chat_visible'):
    st.markdown("""
    <div class="section-container" style="text-align: center; padding: 3.5rem 2rem;">
        <div style="background: var(--rpp-primary-light); width: 80px; height: 80px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 2.5rem; margin-bottom: 1.5rem;">
            💬
        </div>
        <h3 style="color: var(--rpp-dark); margin-bottom: 1rem; font-size: 1.5rem; font-weight: 700;">¿Necesitas analizar los documentos?</h3>
        <p style="color: var(--rpp-gray-600); margin-bottom: 2rem; font-size: 1rem;">Nuestro asistente de IA puede responder preguntas sobre el contenido procesado</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("💬 Activar Asistente de IA", use_container_width=True, type="primary"):
        st.session_state['chat_visible'] = True
        st.rerun()
if st.session_state.get('chat_visible'):
    # Inicializar session ID
    if 'bedrock_session_id' not in st.session_state:
        st.session_state.bedrock_session_id = str(uuid.uuid4())
    
    # Variable para capturar sugerencia clickeada
    if 'pending_suggestion' not in st.session_state:
        st.session_state['pending_suggestion'] = None
    
    # ============================================
    # CONTAINER DEL CHAT - ESTILO COPILOT
    # ============================================
    st.markdown("""
    <div class="chat-container">
        <div class="chat-header">
            <div class="chat-header-content">
                <div class="chat-avatar">🤖</div>
                <div class="chat-header-text">
                    <h2>Asistente de IA</h2>
                    <p>Consulta inteligente sobre tus documentos</p>
                </div>
            </div>
            <div class="chat-status">
                <span class="chat-status-dot"></span>
                <span>IA Activa y Lista</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ============================================
    # SUGERENCIAS INTELIGENTES - CLICKEABLES
    # ============================================
    if st.session_state.get('results') and len(st.session_state['chat_messages']) == 0:
        suggestions = get_chat_suggestions(st.session_state['results'])
        
        suggestion_config = {
            "📊 Comparar proveedores y analizar costos": {"icon": "📊", "desc": "Análisis comparativo de costos y beneficios"},
            "💰 ¿Cuál proveedor es más conveniente?": {"icon": "💰", "desc": "Recomendación basada en datos"},
            "📈 Mostrar tendencias de compra": {"icon": "📈", "desc": "Visualiza patrones y tendencias"},
            "🧾 Resumir montos y fechas de facturas": {"icon": "🧾", "desc": "Resumen ejecutivo de facturas"},
            "⚖️ Analizar cláusulas contractuales": {"icon": "⚖️", "desc": "Revisión legal de contratos"},
            "⚠️ Identificar posibles penalidades": {"icon": "⚠️", "desc": "Detecta riesgos y sanciones"},
            "📑 Revisar obligaciones legales": {"icon": "📑", "desc": "Extrae compromisos legales"},
            "📅 Verificar fechas y plazos importantes": {"icon": "📅", "desc": "Identifica fechas críticas"},
            "📋 Comparar contenido entre documentos": {"icon": "📋", "desc": "Encuentra similitudes y diferencias"},
            "🔍 Encontrar información común": {"icon": "🔍", "desc": "Detecta datos compartidos"},
            "📋 Resumir el contenido principal": {"icon": "📋", "desc": "Síntesis ejecutiva del documento"},
            "🔍 Buscar información específica": {"icon": "🔍", "desc": "Búsqueda inteligente de datos"},
            "📊 Extraer datos importantes": {"icon": "📊", "desc": "Extracción automática de información clave"}
        }
        
        st.markdown("""
        <div class="suggestions-section">
            <div class="suggestions-title">
                <span class="sparkle">✨</span>
                <h3>Sugerencias Inteligentes</h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Renderizar sugerencias como cards clickeables
        cols = st.columns(3)
        for idx, suggestion in enumerate(suggestions):
            config = suggestion_config.get(suggestion, {"icon": "💡", "desc": "Consulta personalizada"})
            suggestion_text = suggestion.split(' ', 1)[1] if ' ' in suggestion else suggestion
            col_idx = idx % 3
            
            with cols[col_idx]:
                # Card clickeable directo (usando form para simular click)
                if st.button(
                    f"{config['icon']} {suggestion_text}",
                    key=f"sugg_{idx}",
                    use_container_width=True,
                    help=config['desc']
                ):
                    st.session_state['pending_suggestion'] = suggestion
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ============================================
    # HISTORIAL DE MENSAJES
    # ============================================
    st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
    
    for msg in st.session_state['chat_messages']:
        if msg['role'] == 'user':
            st.markdown(f"""
            <div class="user-message">
                <div class="user-message-avatar">👤</div>
                <div class="user-message-content">
                    <div class="message-label">Tú</div>
                    <p class="message-text">{msg['content']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="assistant-message">
                <div class="assistant-message-avatar">🤖</div>
                <div class="assistant-message-content">
                    <div class="message-label">Asistente IA</div>
                    <p class="message-text">{msg['content']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ============================================
    # INPUT DE USUARIO
    # ============================================
    st.markdown("""
    <div class="chat-input-section">
        <div class="chat-input-label">
            <span>💭</span>
            <span>Escribe tu consulta personalizada</span>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    user_input = st.chat_input("Pregunta lo que necesites sobre tus documentos...")
    
    # ============================================
    # PROCESAR INPUT O SUGERENCIA
    # ============================================
    query_to_process = None
    
    if user_input:
        query_to_process = user_input
    elif st.session_state.get('pending_suggestion'):
        query_to_process = st.session_state['pending_suggestion']
        st.session_state['pending_suggestion'] = None
    
    if query_to_process:
        st.session_state['chat_messages'].append({'role': 'user', 'content': query_to_process})
        
        with st.spinner("🤖 Analizando con IA..."):
            try:
                # Obtener texto de documentos (limitado para no exceder tokens)
                documents_text = ""
                if st.session_state.get('results'):
                    for idx, result in enumerate(st.session_state['results']):
                        doc_text = result.get('text', '') or ''
                        if doc_text:
                            documents_text += f"\n--- DOCUMENTO {idx+1} ---\n{doc_text[:1000]}"  # Limitar por documento
                
                # 🎯 NUEVO: Usar prompt estructurado
                structured_prompt = get_structured_prompt(
                    query=query_to_process,
                    provider_analysis=st.session_state.get('provider_analysis'),
                    documents_text=documents_text
                )
                
                agent_response = invoke_agent_legacy(
                    agent_id=SUPERVISOR_AGENT_ID,
                    agent_alias_id=SUPERVISOR_ALIAS_ID,
                    session_id=st.session_state.bedrock_session_id,
                    input_text=structured_prompt
                )
                
                # Manejar errores de manera más específica
                if any(error_keyword in agent_response.lower() for error_keyword in ['error', 'exception', 'timeout', 'failed']):
                    agent_response = """
    ## ⚠️ Análisis No Disponible

    No pude completar el análisis en este momento. 

    **Por favor intenta:**
    1. Verificar que los documentos contengan información de proveedores
    2. Procesar nuevamente los documentos
    3. Consultar información específica como "precios de gaseosas" o "mejor proveedor de lácteos"

    *El sistema detectó un error temporal en el procesamiento.*
    """
                
                st.session_state['chat_messages'].append({'role': 'assistant', 'content': agent_response})
                st.rerun()
                
            except Exception as e:
                error_response = f"""
    ## 🔧 Error Técnico

    **Detalle del error:** {str(e)}

    **Solución sugerida:**
    1. Verificar la conexión con los servicios de AWS
    2. Validar que los documentos sean legibles
    3. Intentar con menos documentos simultáneamente

    *Para asistencia inmediata, contacta al equipo técnico.*
    """
                st.session_state['chat_messages'].append({'role': 'assistant', 'content': error_response})
                st.rerun()