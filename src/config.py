"""
config.py
---------
Resolución centralizada de la OPENAI_API_KEY del lado del servidor.

Orden de búsqueda:
  1. st.secrets["OPENAI_API_KEY"]  — Streamlit Cloud (Settings → Secrets)
  2. Variables de entorno / .env    — desarrollo local y scripts CLI

La key pertenece al administrador de la app: nunca se solicita al visitante,
nunca se escribe en el repositorio y nunca se muestra en pantalla ni en logs.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv


def get_openai_api_key() -> str | None:
    """Return the OpenAI API key, or None if not configured anywhere."""
    # 1. Streamlit secrets — solo si hay un runtime de Streamlit activo.
    #    Import lazy + guard para que los módulos src/ funcionen desde CLI
    #    (python -m src.vectorstore, eval/) sin depender de Streamlit.
    try:
        from streamlit import runtime
        import streamlit as st
        if runtime.exists() and "OPENAI_API_KEY" in st.secrets:
            key = str(st.secrets["OPENAI_API_KEY"]).strip()
            if key:
                return key
    except Exception:
        pass  # sin streamlit, sin secrets.toml, o fuera del runtime

    # 2. Entorno / .env (dev local, CLI, eval)
    load_dotenv()
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    return key or None


def require_openai_api_key() -> str:
    """Return the API key or raise with setup instructions."""
    key = get_openai_api_key()
    if not key:
        raise EnvironmentError(
            "OPENAI_API_KEY no configurada. "
            "Local: copia .env.example a .env y añade tu clave. "
            "Streamlit Cloud: añádela en Settings → Secrets."
        )
    return key
