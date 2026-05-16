import streamlit as st
from openai import OpenAI
import json
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
from langdetect import detect

from core.vector_store import store_meeting_notes, get_all_chunks, get_all_metadatas, get_all_embeddings
from utils.tts_utils import text_to_speech_bytes
from utils.ocr_utils import extract_text
from core.chain_builder import build_rag_chain
from data.tools import jira_tools, email_tools
from data.mock_data import SAMPLE_MEETINGS

st.set_page_config(page_title="Meeting AI Assistant", page_icon="📝", layout="wide")


# ------------------------------------------------------------------ #
#  Session state                                                      #
# ------------------------------------------------------------------ #
def init_state():
    defaults = {
        "chat_history": [],
        "context_data": {},
        "app_stage": "intro",
        "uploaded_name": "",
        "uploaded_content": "",
        "assistant_role": "Manager",
        "store_ready": False,
        "rag_chain": None,
        "_chain_role": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #
def make_client(base_url, api_key):
    if not base_url or not api_key:
        return None
    return OpenAI(base_url=base_url, api_key=api_key)


def get_rag_chain(base_url, api_key, role):
    """Return cached chain, rebuilding only if role changed. History is preserved."""
    if st.session_state.rag_chain is None or st.session_state._chain_role != role:
        st.session_state.rag_chain = build_rag_chain(base_url, api_key, role)
        st.session_state._chain_role = role
        # KHÔNG xóa chat_history nữa — history thuộc về session, không phải chain
    return st.session_state.rag_chain


def format_history_for_chain(chat_history):
    """
    Convert UI history [{role, content}, ...] to LangChain format [(human, ai), ...].
    LangChain's ConversationalRetrievalChain expects tuples of (human_msg, ai_msg).
    """
    pairs = []
    pending_user = None
    for msg in chat_history:
        if msg["role"] == "user":
            pending_user = msg["content"]
        elif msg["role"] == "assistant" and pending_user is not None:
            pairs.append((pending_user, msg["content"]))
            pending_user = None
    return pairs


# ------------------------------------------------------------------ #
#  Render: Intro                                                      #
# ------------------------------------------------------------------ #
def render_intro():
    st.title("📝 Meeting AI Assistant")
    st.markdown("### Welcome! Configure your API and upload notes in the sidebar.")
    col1, col2, col3 = st.columns(3)
    col1.info("**Step 1:** Enter Azure OpenAI details.")
    col2.info("**Step 2:** Upload a `.txt`, `.pdf`, or image file.")
    col3.info("**Step 3:** Click 'Generate' to analyze.")


# ------------------------------------------------------------------ #
#  Render: Processing                                                 #
# ------------------------------------------------------------------ #
def render_processing_state(client):
    st.title("⚙️ Processing Meeting Notes")
    st.info(f"Analyzing: **{st.session_state.uploaded_name}**")

    try:
        with st.status("AI is thinking...", expanded=True) as status:

            status.write("🧠 Storing notes in FAISS vector index...")
            store_meeting_notes(st.session_state.uploaded_content, source=st.session_state.uploaded_name)
            st.session_state.store_ready = True
            st.session_state.rag_chain = None  # force rebuild with fresh data

            status.write("📝 Generating executive summary...")
            res_summary = client.chat.completions.create(
                model="GPT-4o",
                messages=[
                    {"role": "system", "content": "Summarize strictly using: ### Overview, ### Key Results, ### Action Items"},
                    {"role": "user", "content": st.session_state.uploaded_content}
                ],
                temperature=0.5
            )
            summary = res_summary.choices[0].message.content

            status.write("🎫 Creating Jira tickets...")
            res_jira = client.chat.completions.create(
                model="GPT-4o",
                messages=[
                    {"role": "system", "content": "You are a Scrum Master. Extract action items and format them as Jira tickets using the provided tool."},
                    {"role": "user", "content": st.session_state.uploaded_content}
                ],
                tools=jira_tools,
                tool_choice={"type": "function", "function": {"name": "create_jira_tickets"}},
                temperature=0.1
            )
            tool_call = res_jira.choices[0].message.tool_calls[0]
            jira_data = json.loads(tool_call.function.arguments)

            markdown_jira = ""
            for ticket in jira_data["tickets"]:
                markdown_jira += f"### {ticket['summary']}\n"
                markdown_jira += f"**Priority:** {ticket['priority']}\n\n"
                markdown_jira += f"{ticket['description']}\n\n"
                markdown_jira += "**Acceptance Criteria:**\n"
                for ac in ticket["acceptance_criteria"]:
                    markdown_jira += f"- {ac}\n"
                markdown_jira += "\n---\n"

            status.write("🧪 Building testing plan...")
            res_testing = client.chat.completions.create(
                model="GPT-4o",
                messages=[
                    {"role": "system", "content": "Generate a testing plan table based on Jira tickets."},
                    {"role": "user", "content": markdown_jira}
                ],
                temperature=0.2
            )

            st.session_state.context_data = {
                "meeting": st.session_state.uploaded_content,
                "summary": summary,
                "jira": markdown_jira,
                "testing": res_testing.choices[0].message.content,
            }
            st.session_state.app_stage = "generated"
            status.update(label="Analysis Complete!", state="complete")

        st.rerun()

    except Exception as e:
        st.error(f"Generation failed: {e}")
        st.session_state.app_stage = "ready"


# ------------------------------------------------------------------ #
#  Render: Outputs                                                    #
# ------------------------------------------------------------------ #
def render_outputs(client, base_url, api_key):
    st.title("✅ Analysis Complete")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Reports", "🤖 AI Chat", "⚡ Actions", "📄 Context"])

    # ---- Tab 1: Reports ----
    with tab1:
        s1, s2, s3, s4, s5 = st.tabs(["Summary", "Jira", "Testing", "🧠 Knowledge Base", "🔵 Embeddings"])

        with s1:
            summary_text = st.session_state.context_data.get("summary", "")
            st.markdown(summary_text)
            st.divider()

            if st.button("🔊 Read Summary Aloud", key="tts_btn"):
                try:
                    with st.spinner("✍️ Simplifying for audio..."):
                        res_spoken = client.chat.completions.create(
                            model="GPT-4o",
                            messages=[
                                {"role": "system", "content": (
                                    "Convert to a spoken version. Rules: under 120 words, "
                                    "natural spoken language, no markdown or lists, flowing sentences."
                                )},
                                {"role": "user", "content": summary_text}
                            ],
                            temperature=0.4
                        )
                        spoken_text = res_spoken.choices[0].message.content

                    try:
                        detected_lang = detect(summary_text)
                    except Exception:
                        detected_lang = "en"

                    with st.expander("📝 Spoken version"):
                        st.write(spoken_text)
                        st.caption(f"Detected language: `{detected_lang}`")

                    with st.spinner("🔊 Generating audio..."):
                        audio_bytes = text_to_speech_bytes(spoken_text, lang=detected_lang)
                    st.audio(audio_bytes, format="audio/mp3")

                except Exception as e:
                    st.warning(f"TTS failed: {e}")

        s2.markdown(st.session_state.context_data.get("jira", ""))
        s3.markdown(st.session_state.context_data.get("testing", ""))

        with s4:
            st.subheader("🧠 FAISS Knowledge Base")
            st.caption("Text chunks stored as FAISS vector embeddings from your uploaded meeting notes.")
            chunks = get_all_chunks()
            metadatas = get_all_metadatas()
            if not chunks:
                st.warning("No data in FAISS index. Run analysis first.")
            else:
                st.success(f"✅ {len(chunks)} chunks stored")
                st.divider()
                for i, (chunk, meta) in enumerate(zip(chunks, metadatas)):
                    label = f"Chunk {i + 1} — {chunk[:60]}..."
                    with st.expander(label):
                        st.caption(f"source: `{meta.get('source', '—')}` | index: `{meta.get('index')}` | chars: `{meta.get('chars')}`")
                        st.text(chunk)

        with s5:
            st.subheader("🔵 Vector Embeddings Visualization")
            st.caption("Each point is a text chunk — points closer together share semantic similarity.")
            chunks, embeddings = get_all_embeddings()

            if len(chunks) == 0 or len(embeddings) == 0:
                st.warning("No embeddings. Run analysis first.")
            elif len(chunks) < 2:
                st.warning("Need at least 2 chunks to visualize.")
            else:
                arr = np.array(embeddings)
                method = st.radio("Dimensionality reduction", ["PCA", "t-SNE"], horizontal=True)

                if method == "t-SNE":
                    if len(chunks) < 3:
                        st.warning("t-SNE requires at least 3 chunks.")
                        st.stop()
                    from sklearn.manifold import TSNE
                    perplexity = max(1.0, min(5.0, len(chunks) - 1))
                    coords = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(arr)
                else:
                    n_components = min(2, arr.shape[0] - 1, arr.shape[1])
                    coords = PCA(n_components=n_components, random_state=42).fit_transform(arr)

                if coords.shape[1] < 2:
                    coords = np.hstack([coords, np.zeros((coords.shape[0], 1))])

                labels = [
                    f"Chunk {i + 1}: {c[:80]}..." if len(c) > 80 else f"Chunk {i + 1}: {c}"
                    for i, c in enumerate(chunks)
                ]
                fig = px.scatter(
                    x=coords[:, 0], y=coords[:, 1],
                    hover_name=labels,
                    text=[f"#{i + 1}" for i in range(len(chunks))],
                    title=f"Embedding Space — {method} 2D ({len(chunks)} chunks)",
                    labels={"x": f"{method}1", "y": f"{method}2"},
                    color=list(range(len(chunks))),
                    color_continuous_scale="Viridis",
                )
                fig.update_traces(textposition="top center", marker=dict(size=12))
                fig.update_layout(coloraxis_showscale=False, height=500)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(f"**Total chunks:** {len(chunks)} | **Vector dim:** {arr.shape[1]}")

    # ---- Tab 2: AI Chat (Langchain RAG) ----
    with tab2:
        st.subheader("💬 Chat Assistant")
        st.caption("Powered by Langchain ConversationalRetrievalChain + FAISS semantic search.")

        role_col, _ = st.columns([1, 3])
        with role_col:
            role = st.selectbox("Assistant Lens", ["Manager", "Developer", "QA"], key="role_selector")

        st.divider()
        chat_container = st.container()

        with chat_container:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        if prompt := st.chat_input("Ask about the meeting..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    try:
                        chain = get_rag_chain(base_url, api_key, role)
                        # Lấy history TRƯỚC khi append prompt mới (history = các turn đã hoàn thành)
                        history_pairs = format_history_for_chain(
                            st.session_state.chat_history[:-1]  # bỏ message user vừa append
                        )
                        with st.spinner("🤖 Thinking..."):
                            result = chain.invoke({
                                "question": prompt,
                                "chat_history": history_pairs,
                            })
                        reply = result["answer"]
                        st.markdown(reply)
                        st.session_state.chat_history.append({"role": "assistant", "content": reply})
                    except Exception as e:
                        st.error(f"Chat failed: {e}")

    # ---- Tab 3: Actions (Function Calling) ----
    with tab3:
        st.subheader("⚡ Quick Actions")

        if st.button("📧 Generate Follow-up Email", use_container_width=True):
            context = st.session_state.context_data
            content = (
                f"Meeting Summary:\n{context.get('summary', '')}\n\n"
                f"Action Items (Jira):\n{context.get('jira', '')}"
            )
            try:
                with st.spinner("✍️ Drafting email..."):
                    res = client.chat.completions.create(
                        model="GPT-4o",
                        messages=[
                            {"role": "system", "content": "Generate a professional follow-up email from the meeting summary and action items."},
                            {"role": "user", "content": content}
                        ],
                        tools=email_tools,
                        tool_choice={"type": "function", "function": {"name": "generate_follow_up_email"}},
                        temperature=0.3
                    )

                tool_call = res.choices[0].message.tool_calls[0]
                email_data = json.loads(tool_call.function.arguments)

                st.divider()
                st.markdown(f"**To:** {email_data['recipients']}")
                st.markdown(f"**Subject:** {email_data['subject']}")
                st.divider()
                st.markdown(email_data["body"])
                st.markdown("**Action Items:**")
                for item in email_data["action_items"]:
                    st.markdown(f"- {item}")
                st.markdown(f"\n{email_data['closing']}")

            except Exception as e:
                st.error(f"Email generation failed: {e}")

    # ---- Tab 4: Raw Context ----
    with tab4:
        st.json(st.session_state.context_data)


# ------------------------------------------------------------------ #
#  Sidebar                                                            #
# ------------------------------------------------------------------ #
with st.sidebar:
    st.header("⚙️ Configuration")
    base_url = st.text_input("Azure Endpoint", value="")
    api_key = st.text_input("API Key", type="password")

    st.divider()

    st.subheader("📂 Load Sample Meeting")
    sample_choice = st.selectbox("Choose a sample", ["— Select —"] + list(SAMPLE_MEETINGS.keys()))
    if st.button("Load Sample", disabled=(sample_choice == "— Select —")):
        st.session_state.chat_history = []
        st.session_state.context_data = {}
        st.session_state.store_ready = False
        st.session_state.rag_chain = None
        st.session_state.uploaded_content = SAMPLE_MEETINGS[sample_choice]
        st.session_state.uploaded_name = sample_choice
        st.session_state.app_stage = "ready"
        st.rerun()

    st.divider()

    uploaded_file = st.file_uploader(
        "Or Upload Notes",
        type=["txt", "pdf", "jpg", "jpeg", "png"],
        help="Supports: .txt, .pdf, .jpg, .png"
    )

    if uploaded_file:
        raw_bytes = uploaded_file.read()
        filename = uploaded_file.name
        ext = filename.lower().split(".")[-1]

        if filename != st.session_state.get("uploaded_name", ""):
            st.session_state.chat_history = []
            st.session_state.context_data = {}
            st.session_state.store_ready = False
            st.session_state.rag_chain = None
            st.session_state.app_stage = "ready"
            st.info("🔄 New file detected — conversation reset.")

        if ext == "txt":
            content = raw_bytes.decode("utf-8")
        else:
            with st.spinner(f"🔍 Running OCR on {filename}..."):
                content = extract_text(raw_bytes, filename)
            with st.expander("👁️ OCR Preview"):
                st.text(content[:500] + ("..." if len(content) > 500 else ""))

        st.session_state.uploaded_content = content
        st.session_state.uploaded_name = filename

        if st.session_state.app_stage == "intro":
            st.session_state.app_stage = "ready"

    st.divider()

    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
        if not base_url or not api_key or not st.session_state.uploaded_content:
            st.error("Missing config or file!")
        else:
            st.session_state.app_stage = "processing"
            st.rerun()

    if st.button("🔄 Reset App"):
        st.session_state.clear()
        st.rerun()


# ------------------------------------------------------------------ #
#  App router                                                         #
# ------------------------------------------------------------------ #
client = make_client(base_url, api_key)

if st.session_state.app_stage == "intro":
    render_intro()
elif st.session_state.app_stage == "ready":
    st.title("📂 File Loaded")
    st.success(f"Ready to process: {st.session_state.uploaded_name}")
    st.text_area("Preview", st.session_state.uploaded_content, height=300)
elif st.session_state.app_stage == "processing":
    render_processing_state(client)
elif st.session_state.app_stage == "generated":
    render_outputs(client, base_url, api_key)
