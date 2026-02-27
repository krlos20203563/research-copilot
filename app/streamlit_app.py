"""
streamlit_app.py
--------
Research Copilot — RAG interface for 20 academic papers on criminal governance
and extortion in Latin America.

The app requires an OpenAI API key entered at runtime. The key is stored only
in st.session_state (memory) and is never written to disk or source code.

Run with:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st

# Make project root importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="Research Copilot — Crimen Organizado & Gobernanza Criminal",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ─────────────────────────────────────────────────
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "api_key_validated" not in st.session_state:
    st.session_state.api_key_validated = False
if "messages" not in st.session_state:
    st.session_state.messages = []


# ── API Key gate ───────────────────────────────────────────────────────────

def _validate_api_key(key: str) -> tuple[bool, str]:
    """
    Do a lightweight check: verify the key is non-empty and well-formed,
    then make a minimal API call to confirm it works.
    Returns (is_valid, error_message).
    """
    if not key or not key.strip():
        return False, "La clave no puede estar vacía."
    key = key.strip()
    if not key.startswith("sk-"):
        return False, "Una API key de OpenAI debe comenzar con 'sk-'."
    try:
        from openai import OpenAI, AuthenticationError
        client = OpenAI(api_key=key)
        # Minimal call — list models is cheap and fast
        client.models.list()
        return True, ""
    except AuthenticationError:
        return False, "API key inválida. Verifica que la clave es correcta."
    except Exception as exc:
        return False, f"Error al verificar la clave: {exc}"


def render_api_key_gate() -> bool:
    """
    Show the full-page API key input screen.
    Returns True only when a valid key is stored in session state.
    """
    if st.session_state.api_key_validated:
        return True

    # Center the form
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown("## 📚 Research Copilot")
        st.markdown(
            "Asistente de investigación para 20 artículos académicos sobre "
            "**crimen organizado, extorsión y gobernanza criminal** en América Latina."
        )
        st.divider()
        st.markdown("### 🔑 Ingresa tu OpenAI API Key")
        st.caption(
            "La clave se almacena únicamente en memoria durante esta sesión "
            "y nunca se guarda en disco ni en el código fuente. "
            "Obtén la tuya en [platform.openai.com/api-keys](https://platform.openai.com/api-keys)."
        )

        with st.form("api_key_form", clear_on_submit=False):
            key_input = st.text_input(
                "API Key",
                type="password",
                placeholder="sk-...",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button(
                "Iniciar Research Copilot →",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            with st.spinner("Verificando clave…"):
                valid, error_msg = _validate_api_key(key_input)

            if valid:
                st.session_state.api_key = key_input.strip()
                st.session_state.api_key_validated = True
                st.rerun()
            else:
                st.error(f"❌ {error_msg}")

        st.divider()
        st.caption(
            "💡 **Para desarrollo local:** crea un archivo `.env` con "
            "`OPENAI_API_KEY=sk-...` y la app lo cargará automáticamente."
        )

    return False


# ── Try loading key from .env for local development (never from source code) ─
def _try_load_env_key():
    """
    Load key from .env if present and not already set.
    This only runs once at startup. The key is stored in session state only.
    """
    if st.session_state.api_key_validated:
        return
    try:
        from dotenv import dotenv_values
        env = dotenv_values(ROOT / ".env")
        key = env.get("OPENAI_API_KEY", "").strip()
        if key and key != "sk-...your-key-here...":
            st.session_state.api_key = key
            st.session_state.api_key_validated = True
    except Exception:
        pass


# ── Cached resources ───────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Conectando con ChromaDB…")
def get_chroma_client():
    from src.vectorstore import get_chroma_client as _get
    return _get()


@st.cache_data(show_spinner="Cargando metadatos de papers…")
def load_papers_metadata():
    json_path = ROOT / "papers" / "papers.json"
    if not json_path.exists():
        return []
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("papers", [])


def get_openai_client():
    """Return an OpenAI client using the session-state key."""
    from openai import OpenAI
    return OpenAI(api_key=st.session_state.api_key)


# ── Sidebar ────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.title("⚙️ Configuración")

        strategy = st.selectbox(
            "Estrategia de prompts",
            options=["delimiters", "json", "fewshot", "cot"],
            format_func=lambda x: {
                "delimiters": "1 — Delimitadores",
                "json": "2 — Salida JSON",
                "fewshot": "3 — Few-Shot",
                "cot": "4 — Chain-of-Thought",
            }[x],
            key="strategy",
        )

        chunk_strategy = st.radio(
            "Tamaño de chunks",
            options=["small", "large"],
            format_func=lambda x: "256 tokens" if x == "small" else "1024 tokens",
            horizontal=True,
            key="chunk_strategy",
        )

        top_k = st.slider("Top-k fragmentos", 1, 10, 5, key="top_k")

        st.divider()

        # Show masked key indicator + logout
        masked = "sk-…" + st.session_state.api_key[-4:] if len(st.session_state.api_key) > 6 else "—"
        st.caption(f"🔑 API Key: `{masked}`")
        if st.button("🔒 Cerrar sesión / Cambiar clave", use_container_width=True):
            st.session_state.api_key = ""
            st.session_state.api_key_validated = False
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.caption("Research Copilot v0.1.0")

    return strategy, chunk_strategy, top_k


# ── Tab 1: Chat ────────────────────────────────────────────────────────────

def render_chat_tab(strategy: str, chunk_strategy: str, top_k: int):
    st.header("💬 Chat con los Papers")
    st.caption(
        "Haz preguntas sobre crimen organizado, extorsión y gobernanza criminal "
        "en América Latina. Las respuestas se basan en los 20 artículos indexados."
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 Fuentes consultadas", expanded=False):
                    for src in msg["sources"]:
                        st.markdown(
                            f"**{src['title']}** — {', '.join(src['authors'][:2])} ({src['year']})  \n"
                            f"Relevancia: `{src['score']:.3f}` | {src.get('venue', '')}"
                        )

    if prompt := st.chat_input("¿Cuál es tu pregunta de investigación?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Buscando en los papers…"):
                answer, sources = _run_rag(prompt, strategy, chunk_strategy, top_k)
            st.markdown(answer)
            if sources:
                with st.expander("📎 Fuentes consultadas", expanded=True):
                    for src in sources:
                        st.markdown(
                            f"**{src['title']}** — {', '.join(src['authors'][:2])} ({src['year']})  \n"
                            f"Relevancia: `{src['score']:.3f}` | {src.get('venue', '')}"
                        )

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )

    if st.session_state.messages:
        if st.button("🗑️ Limpiar conversación", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()


def _run_rag(question: str, strategy: str, chunk_strategy: str, top_k: int):
    from src.retrieval import search
    from src.generation import generate_answer

    chroma = get_chroma_client()
    oa = get_openai_client()

    try:
        chunks = search(
            question,
            top_k=top_k,
            strategy=chunk_strategy,
            chroma_client=chroma,
            openai_client=oa,
        )
    except RuntimeError as e:
        return (
            f"⚠️ El índice no está construido. Ejecuta primero:\n\n"
            f"```\npython -m src.vectorstore\n```\n\nError: {e}",
            [],
        )

    result = generate_answer(question=question, chunks=chunks, strategy=strategy, client=oa)
    sources = [
        {"title": c.title, "authors": c.authors, "year": c.year,
         "venue": c.venue, "score": c.score}
        for c in chunks
    ]
    return result["answer"], sources


# ── Tab 2: Paper Browser ───────────────────────────────────────────────────

def render_papers_tab():
    st.header("📄 Explorador de Papers")
    papers = load_papers_metadata()
    if not papers:
        st.warning("No se encontró papers.json.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        years = sorted({p["year"] for p in papers if p.get("year")})
        selected_years = st.multiselect("Año", options=years)
    with col2:
        all_topics = sorted({t for p in papers for t in (p.get("topics") or [])})
        selected_topics = st.multiselect("Tema", options=all_topics)
    with col3:
        search_text = st.text_input("Buscar por título / autor", "")

    filtered = papers
    if selected_years:
        filtered = [p for p in filtered if p.get("year") in selected_years]
    if selected_topics:
        filtered = [p for p in filtered if any(t in (p.get("topics") or []) for t in selected_topics)]
    if search_text:
        q = search_text.lower()
        filtered = [
            p for p in filtered
            if q in p.get("title", "").lower()
            or any(q in a.lower() for a in (p.get("authors") or []))
        ]

    st.caption(f"Mostrando {len(filtered)} de {len(papers)} papers")

    for paper in filtered:
        authors_str = "; ".join((paper.get("authors") or [])[:3])
        if len(paper.get("authors") or []) > 3:
            authors_str += " et al."

        with st.expander(
            f"**{paper.get('title', '?')}** — {authors_str} ({paper.get('year', '?')})"
        ):
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.markdown(f"**Autores:** {authors_str}")
                st.markdown(f"**Venue:** {paper.get('venue') or '—'}")
                if paper.get("doi"):
                    st.markdown(f"**DOI:** [{paper['doi']}](https://doi.org/{paper['doi']})")
                if paper.get("abstract"):
                    st.markdown("**Abstract:**")
                    st.caption(paper["abstract"])
            with col_b:
                for t in (paper.get("topics") or []):
                    st.markdown(f"- {t}")


# ── Tab 3: Compare Strategies ─────────────────────────────────────────────

def render_compare_tab(chunk_strategy: str, top_k: int):
    st.header("🔬 Comparar Estrategias de Prompts")
    st.caption("Compara las 4 estrategias de prompting sobre la misma pregunta.")

    question = st.text_area(
        "Pregunta:",
        value="¿Cuál es la relación entre gobernanza criminal y el Estado en América Latina?",
        height=80,
        key="compare_question",
    )

    if st.button("▶ Ejecutar las 4 estrategias", type="primary", key="run_compare"):
        from src.retrieval import search
        from src.generation import generate_answer, STRATEGY_LABELS

        chroma = get_chroma_client()
        oa = get_openai_client()

        with st.spinner("Recuperando fragmentos…"):
            try:
                chunks = search(question, top_k=top_k, strategy=chunk_strategy,
                                chroma_client=chroma, openai_client=oa)
            except RuntimeError as e:
                st.error(f"Index not built: {e}")
                return

        st.success(f"Recuperados {len(chunks)} fragmentos.")
        with st.expander("📎 Fragmentos recuperados", expanded=False):
            for c in chunks:
                st.markdown(f"**[{c.score:.3f}]** {c.title} ({c.year}) — chunk {c.chunk_index}")
                st.caption(c.text[:300] + "…")

        cols = st.columns(2)
        for i, strat in enumerate(["delimiters", "json", "fewshot", "cot"]):
            with cols[i % 2]:
                st.subheader(STRATEGY_LABELS[strat])
                with st.spinner(f"Generando con {strat}…"):
                    res = generate_answer(question=question, chunks=chunks,
                                         strategy=strat, client=oa)
                st.markdown(res["answer"])
                st.caption(f"Tokens: {res['total_tokens']} | Tiempo: {res['elapsed_seconds']}s")
                st.divider()


# ── Tab 4: About ───────────────────────────────────────────────────────────

def render_about_tab():
    st.header("ℹ️ Acerca del Research Copilot")
    papers = load_papers_metadata()

    st.markdown("""
## Arquitectura

```
PDF Papers → PyMuPDF → Chunking (256 / 1024 tokens) → text-embedding-3-small
                                                              ↓
                                                       ChromaDB (cosine)
                                                              ↓
User Query → embed → Top-K Retrieval → Prompt Strategy → GPT-4o → Answer
```

## Estrategias de Prompting

| # | Estrategia | Descripción |
|---|-----------|-------------|
| 1 | **Delimitadores** | Secciones XML (`<<<CONTEXTO>>>`, `<<<PREGUNTA>>>`) |
| 2 | **JSON Output** | Respuesta estructurada con campos predefinidos |
| 3 | **Few-Shot** | Dos ejemplos trabajados enseñan el estilo esperado |
| 4 | **Chain-of-Thought** | 5 pasos explícitos de razonamiento antes de responder |

## Seguridad

- La API key **nunca** se almacena en el código fuente ni en archivos versionados.
- Se solicita al usuario al inicio de cada sesión y se guarda solo en memoria.
- El repositorio es seguro para hacerse público en GitHub.

## Papers indexados
""")

    if papers:
        for p in papers:
            authors = (p.get("authors") or [])[:2]
            st.markdown(f"- **{p.get('title', '?')}** — {'; '.join(authors)} ({p.get('year', '?')})")

    st.markdown("""
## Uso local

```bash
git clone <repo>
cd Tarea_1
pip install -r requirements.txt

# Indexar papers (una sola vez)
python -m src.vectorstore

# Lanzar la app
streamlit run app/streamlit_app.py
```

La app pedirá tu API key en el navegador.
""")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    # 1. Try loading key from local .env (dev convenience — never committed)
    _try_load_env_key()

    # 2. Gate: show API key input if not authenticated
    if not render_api_key_gate():
        st.stop()

    # 3. Main app
    strategy, chunk_strategy, top_k = render_sidebar()

    tab_chat, tab_papers, tab_compare, tab_about = st.tabs([
        "💬 Chat", "📄 Papers", "🔬 Comparar", "ℹ️ Acerca de",
    ])

    with tab_chat:
        render_chat_tab(strategy, chunk_strategy, top_k)
    with tab_papers:
        render_papers_tab()
    with tab_compare:
        render_compare_tab(chunk_strategy, top_k)
    with tab_about:
        render_about_tab()


if __name__ == "__main__":
    main()
