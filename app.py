"""
🎓 Умный суммаризатор и генератор учебных материалов
Streamlit + Anthropic Claude API
"""

import json
import os
import random
import re
from io import BytesIO

import anthropic
import streamlit as st
from pypdf import PdfReader

# ----------------------------------------------------------------------------
# Настройка страницы и стили
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Умный учебный помощник",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1100px;}
    .hero {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
        padding: 1.6rem 2rem; border-radius: 18px; color: white; margin-bottom: 1.2rem;
        box-shadow: 0 8px 24px rgba(99,102,241,.25);
    }
    .hero h1 {margin: 0; font-size: 2rem; color: white;}
    .hero p {margin: .3rem 0 0 0; opacity: .92;}
    .thesis {
        background: rgba(99,102,241,.08); border-left: 5px solid #6366f1;
        padding: .8rem 1rem; border-radius: 10px; margin-bottom: .6rem;
    }
    .term {
        background: rgba(236,72,153,.07); border-left: 5px solid #ec4899;
        padding: .7rem 1rem; border-radius: 10px; margin-bottom: .5rem;
    }
    .flashcard {
        min-height: 190px; display: flex; align-items: center; justify-content: center;
        text-align: center; padding: 2rem; border-radius: 20px; font-size: 1.3rem;
        font-weight: 500; box-shadow: 0 6px 20px rgba(0,0,0,.12); margin: .8rem 0 1rem 0;
    }
    .fc-q {background: linear-gradient(135deg,#dbeafe,#e0e7ff); color: #1e1b4b;}
    .fc-a {background: linear-gradient(135deg,#dcfce7,#d1fae5); color: #052e16;}
    .score-box {
        text-align: center; padding: 1.5rem; border-radius: 18px;
        background: linear-gradient(135deg,#f59e0b,#ef4444); color: white; margin-top: 1rem;
    }
    .score-box h2 {color: white; margin: 0; font-size: 2.4rem;}

    /* ---------- Мобильная адаптация ---------- */
    .stButton > button {min-height: 2.8rem; border-radius: 12px;}
    div[data-baseweb="tab-list"] {overflow-x: auto; scrollbar-width: none;}
    div[data-baseweb="tab-list"]::-webkit-scrollbar {display: none;}
    @media (max-width: 640px) {
        .block-container {padding: 1rem .8rem 4rem .8rem;}
        .hero {padding: 1rem 1.1rem; border-radius: 14px;}
        .hero h1 {font-size: 1.35rem;}
        .hero p {font-size: .85rem;}
        .flashcard {min-height: 150px; padding: 1.2rem; font-size: 1.05rem;}
        .score-box h2 {font-size: 1.8rem;}
        .stTabs [data-baseweb="tab"] {padding: .5rem .6rem; font-size: .85rem;}
        /* держим ряды кнопок/метрик в одну строку */
        div[data-testid="stHorizontalBlock"] {flex-wrap: nowrap !important; gap: .4rem !important;}
        div[data-testid="stHorizontalBlock"] > div {min-width: 0 !important; flex: 1 1 0 !important;}
        div[data-testid="stHorizontalBlock"] .stButton > button {padding: .3rem .2rem; font-size: .8rem;}
        div[data-testid="stMetricValue"] {font-size: 1.05rem;}
        div[data-testid="stMetricLabel"] {font-size: .7rem;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

MODELS = {
    "Claude Sonnet 5.5 (рекомендуется)": "claude-sonnet-5-5",
    "Claude Haiku 5.5 (быстро и дёшево)": "claude-haiku-5-5",
    "Claude Opus 5.5 (максимум качества)": "claude-opus-5-5",
}

# ----------------------------------------------------------------------------
# Состояние сессии
# ----------------------------------------------------------------------------
DEFAULTS = {
    "source_text": "",
    "source_name": "",
    "summary": None,
    "flashcards": None,
    "quiz": None,
    "quiz_attempt": 0,
    "fc_idx": 0,
    "fc_show_answer": False,
    "chat_history": [],
}
for _k, _v in DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def reset_generated():
    st.session_state.summary = None
    st.session_state.flashcards = None
    st.session_state.quiz = None
    st.session_state.quiz_attempt += 1
    st.session_state.fc_idx = 0
    st.session_state.fc_show_answer = False
    st.session_state.chat_history = []


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Настройки")

    api_key = st.text_input(
        "🔑 Anthropic API Key",
        type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        placeholder="sk-ant-...",
        help="Ключ можно получить на console.anthropic.com. Он не сохраняется на диск.",
    )
    model_label = st.selectbox("🤖 Модель", list(MODELS.keys()), index=0)
    model_id = MODELS[model_label]
    custom_model = st.text_input("Другая модель (необязательно)", placeholder="ID модели")
    if custom_model.strip():
        model_id = custom_model.strip()

    st.divider()
    language = st.selectbox("🌍 Язык ответов", ["Русский", "English", "Қазақша", "Кыргызча", "Українська"], index=0)
    level = st.select_slider(
        "🎚️ Уровень объяснений",
        options=["Простой (школа)", "Средний (колледж)", "Продвинутый (вуз)"],
        value="Средний (колледж)",
    )
    n_cards = st.slider("🃏 Количество карточек", 5, 10, 8)
    quiz_difficulty = st.select_slider("🧠 Сложность теста", options=["Лёгкий", "Средний", "Сложный"], value="Средний")
    max_chars = st.slider(
        "📏 Макс. длина текста (символов)", 10_000, 400_000, 150_000, step=10_000,
        help="Слишком длинный текст будет обрезан, чтобы уложиться в контекст модели.",
    )

    st.divider()
    if st.button("🗑️ Очистить всё", use_container_width=True):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v if not isinstance(v, (list, dict)) else type(v)()
        st.rerun()
    st.caption("Сделано на Streamlit + Claude ✨")


# ----------------------------------------------------------------------------
# Вспомогательные функции
# ----------------------------------------------------------------------------
def extract_pdf_text(file) -> str:
    reader = PdfReader(BytesIO(file.getvalue()))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            raise ValueError("PDF защищён паролем.")
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n\n".join(pages).strip()


def get_client():
    if not api_key.strip():
        st.error("🔑 Введите Anthropic API Key в боковой панели.")
        st.stop()
    return anthropic.Anthropic(api_key=api_key.strip())


def doc_text() -> str:
    return st.session_state.source_text[:max_chars]


def base_system(extra: str = "") -> str:
    return (
        "Ты — внимательный и дружелюбный репетитор, помогающий школьникам и студентам "
        f"разбираться в учебных материалах. Отвечай на языке: {language}. "
        f"Уровень объяснений: {level}. Опирайся ТОЛЬКО на предоставленный материал, "
        "ничего не выдумывай. " + extra
    )


def parse_json(raw: str):
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def call_json(task_prompt: str, max_tokens: int = 4000):
    """Запрос к Claude с гарантированным JSON (с одной повторной попыткой)."""
    client = get_client()
    system = base_system("Возвращай ТОЛЬКО валидный JSON без markdown-обёрток и пояснений.")
    user = f"<document>\n{doc_text()}\n</document>\n\n{task_prompt}"
    last_err = None
    for _ in range(2):
        try:
            resp = client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            text = "".join(b.text for b in resp.content if b.type == "text")
            return parse_json(text)
        except json.JSONDecodeError as e:
            last_err = e
            continue
        except anthropic.AuthenticationError:
            st.error("❌ Неверный API-ключ.")
            st.stop()
        except anthropic.RateLimitError:
            st.error("⏳ Превышен лимит запросов. Попробуйте чуть позже.")
            st.stop()
        except anthropic.APIError as e:
            st.error(f"⚠️ Ошибка API: {e}")
            st.stop()
    st.error(f"Не удалось получить корректный JSON от модели: {last_err}")
    st.stop()


def require_text():
    if not st.session_state.source_text.strip():
        st.info("📥 Сначала загрузите PDF или вставьте текст во вкладке «Материал».")
        return False
    return True


# ----------------------------------------------------------------------------
# Генераторы
# ----------------------------------------------------------------------------
def generate_summary():
    prompt = """Составь выжимку материала. Верни JSON строго такой структуры:
{
  "theses": ["ровно 5 главных тезисов/выводов, каждый 1-2 предложения"],
  "terms": [{"term": "термин", "definition": "понятное определение простыми словами"}]
}
В "terms" включи 6-12 ключевых терминов."""
    data = call_json(prompt, 3000)
    st.session_state.summary = {
        "theses": [str(t) for t in data.get("theses", [])][:5],
        "terms": [t for t in data.get("terms", []) if "term" in t and "definition" in t],
    }


def generate_flashcards():
    prompt = f"""Создай ровно {n_cards} карточек для запоминания. Верни JSON:
{{"cards": [{{"question": "...", "answer": "..."}}]}}
Вопросы — чёткие, ответы — краткие (1-3 предложения)."""
    data = call_json(prompt, 3000)
    cards = [c for c in data.get("cards", []) if "question" in c and "answer" in c]
    st.session_state.flashcards = cards
    st.session_state.fc_idx = 0
    st.session_state.fc_show_answer = False


def generate_quiz():
    prompt = f"""Создай тест из ровно 5 вопросов с выбором ответа. Сложность: {quiz_difficulty}.
Верни JSON:
{{"questions": [{{
  "question": "текст вопроса",
  "options": ["вариант A", "вариант B", "вариант C", "вариант D"],
  "correct": 0,
  "explanation": "почему этот ответ верный"
}}]}}
Правила: ровно 4 варианта в "options" (без букв в начале), "correct" — индекс верного (0-3),
правильные ответы должны быть равномерно распределены по позициям, неверные варианты правдоподобны."""
    data = call_json(prompt, 4000)
    valid = []
    for q in data.get("questions", []):
        opts = q.get("options", [])
        corr = q.get("correct")
        if isinstance(corr, str) and corr.strip().upper() in "ABCD" and corr.strip():
            corr = "ABCD".index(corr.strip().upper())
        if len(opts) == 4 and isinstance(corr, int) and 0 <= corr <= 3 and "question" in q:
            valid.append(
                {
                    "question": q["question"],
                    "options": [str(o) for o in opts],
                    "correct": corr,
                    "explanation": q.get("explanation", ""),
                }
            )
    st.session_state.quiz = valid[:5]
    st.session_state.quiz_attempt += 1


# ----------------------------------------------------------------------------
# Шапка и вкладки
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>🎓 Умный учебный помощник</h1>
      <p>Загрузи конспект или учебник — получи тезисы, карточки, тест и чат с документом.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_input, tab_summary, tab_cards, tab_quiz, tab_chat = st.tabs(
    ["📥 Материал", "📝 Выжимка", "🃏 Карточки", "🧠 Тест", "💬 Чат с документом"]
)

# ------------------------------- Материал ------------------------------------
with tab_input:
    st.subheader("Откуда берём материал?")
    method = st.radio("Способ ввода", ["📄 Загрузить PDF", "✍️ Вставить текст"], horizontal=True)

    if method == "📄 Загрузить PDF":
        pdf = st.file_uploader("Выберите PDF-файл", type=["pdf"])
        if pdf is not None and st.button("📖 Извлечь текст", type="primary"):
            with st.spinner("Читаю PDF..."):
                try:
                    text = extract_pdf_text(pdf)
                except Exception as e:
                    st.error(f"Не удалось прочитать PDF: {e}")
                    text = ""
            if not text:
                st.warning("Текст не найден. Возможно, это скан — нужен PDF с текстовым слоем.")
            else:
                st.session_state.source_text = text
                st.session_state.source_name = pdf.name
                reset_generated()
                st.success(f"✅ Извлечено {len(text):,} символов из «{pdf.name}»")
    else:
        pasted = st.text_area(
            "Вставьте текст", height=300, placeholder="Скопируйте сюда текст из методички, учебника или статьи..."
        )
        if st.button("✅ Использовать этот текст", type="primary"):
            if len(pasted.strip()) < 100:
                st.warning("Текст слишком короткий (минимум ~100 символов).")
            else:
                st.session_state.source_text = pasted.strip()
                st.session_state.source_name = "Вставленный текст"
                reset_generated()
                st.success(f"✅ Принято {len(pasted.strip()):,} символов")

    if st.session_state.source_text:
        total = len(st.session_state.source_text)
        c1, c2, c3 = st.columns(3)
        c1.metric("📚 Источник", st.session_state.source_name)
        c2.metric("🔤 Символов", f"{total:,}")
        c3.metric("⏱️ Чтение ≈", f"{max(1, total // 1200)} мин")
        if total > max_chars:
            st.warning(f"Текст будет обрезан до {max_chars:,} символов (см. настройки).")
        with st.expander("👀 Предпросмотр текста"):
            st.text(st.session_state.source_text[:3000] + ("…" if total > 3000 else ""))

        st.divider()
        if st.button("🚀 Сгенерировать всё сразу", use_container_width=True):
            with st.status("Готовлю материалы...", expanded=True) as status:
                st.write("📝 Выжимка...")
                generate_summary()
                st.write("🃏 Карточки...")
                generate_flashcards()
                st.write("🧠 Тест...")
                generate_quiz()
                status.update(label="Готово! Переходите по вкладкам 🎉", state="complete")

# ------------------------------- Выжимка -------------------------------------
with tab_summary:
    if require_text():
        if st.button("📝 Сгенерировать выжимку" if not st.session_state.summary else "🔄 Пересоздать выжимку"):
            with st.spinner("Анализирую материал..."):
                generate_summary()
        s = st.session_state.summary
        if s:
            st.subheader("⭐ 5 главных тезисов")
            for i, t in enumerate(s["theses"], 1):
                st.markdown(f'<div class="thesis"><b>{i}.</b> {t}</div>', unsafe_allow_html=True)
            st.subheader("📖 Ключевые термины")
            for t in s["terms"]:
                st.markdown(
                    f'<div class="term"><b>{t["term"]}</b> — {t["definition"]}</div>',
                    unsafe_allow_html=True,
                )

# ------------------------------- Карточки ------------------------------------
with tab_cards:
    if require_text():
        if st.button("🃏 Создать карточки" if not st.session_state.flashcards else "🔄 Создать новые карточки"):
            with st.spinner("Создаю карточки..."):
                generate_flashcards()
        cards = st.session_state.flashcards
        if cards:
            mode = st.radio("Режим", ["🎴 Тренировка", "📋 Список"], horizontal=True)
            if mode == "🎴 Тренировка":
                idx = min(st.session_state.fc_idx, len(cards) - 1)
                card = cards[idx]
                st.progress((idx + 1) / len(cards), text=f"Карточка {idx + 1} из {len(cards)}")
                if st.session_state.fc_show_answer:
                    st.markdown(f'<div class="flashcard fc-a">💡 {card["answer"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="flashcard fc-q">❓ {card["question"]}</div>', unsafe_allow_html=True)

                b1, b2, b3, b4 = st.columns(4)
                if b1.button("⬅️ Назад", disabled=idx == 0, use_container_width=True):
                    st.session_state.fc_idx = idx - 1
                    st.session_state.fc_show_answer = False
                    st.rerun()
                if b2.button("🔄 Перевернуть", type="primary", use_container_width=True):
                    st.session_state.fc_show_answer = not st.session_state.fc_show_answer
                    st.rerun()
                if b3.button("Вперёд ➡️", disabled=idx == len(cards) - 1, use_container_width=True):
                    st.session_state.fc_idx = idx + 1
                    st.session_state.fc_show_answer = False
                    st.rerun()
                if b4.button("🔀 Перемешать", use_container_width=True):
                    random.shuffle(st.session_state.flashcards)
                    st.session_state.fc_idx = 0
                    st.session_state.fc_show_answer = False
                    st.rerun()
            else:
                for i, c in enumerate(cards, 1):
                    with st.expander(f"❓ {i}. {c['question']}"):
                        st.success(c["answer"])

# ------------------------------- Тест ----------------------------------------
with tab_quiz:
    if require_text():
        if st.button("🧠 Создать тест" if not st.session_state.quiz else "🔄 Создать новый тест"):
            with st.spinner("Придумываю вопросы..."):
                generate_quiz()
        quiz = st.session_state.quiz
        if quiz:
            attempt = st.session_state.quiz_attempt
            st.caption("Выберите вариант — проверка произойдёт мгновенно.")
            score, answered = 0, 0
            for i, q in enumerate(quiz):
                st.markdown(f"#### {i + 1}. {q['question']}")
                labels = [f"{'ABCD'[j]}. {opt}" for j, opt in enumerate(q["options"])]
                choice = st.radio(
                    "Ответ", labels, index=None, key=f"quiz_{attempt}_{i}", label_visibility="collapsed"
                )
                if choice is not None:
                    answered += 1
                    picked = labels.index(choice)
                    if picked == q["correct"]:
                        score += 1
                        st.success("✅ Верно!")
                    else:
                        st.error(f"❌ Неверно. Правильный ответ: **{labels[q['correct']]}**")
                    if q["explanation"]:
                        st.info(f"💡 {q['explanation']}")
                st.divider()

            st.progress(answered / len(quiz), text=f"Отвечено: {answered} из {len(quiz)}")
            if answered == len(quiz):
                pct = round(score / len(quiz) * 100)
                verdict = (
                    "🏆 Великолепно!" if pct == 100
                    else "👏 Отлично!" if pct >= 80
                    else "👍 Хорошо, но есть что повторить" if pct >= 60
                    else "📚 Стоит перечитать материал"
                )
                st.markdown(
                    f'<div class="score-box"><h2>{score} / {len(quiz)} ({pct}%)</h2><p>{verdict}</p></div>',
                    unsafe_allow_html=True,
                )
                if pct == 100:
                    st.balloons()
            if st.button("🔁 Пройти заново"):
                st.session_state.quiz_attempt += 1
                st.rerun()

# ------------------------------- Чат -----------------------------------------
with tab_chat:
    if require_text():
        top = st.columns([4, 1])
        top[0].caption("Задавайте любые уточняющие вопросы — отвечаю только по загруженному тексту.")
        if top[1].button("🧹 Очистить чат"):
            st.session_state.chat_history = []
            st.rerun()

        for m in st.session_state.chat_history:
            with st.chat_message(m["role"], avatar="🧑‍🎓" if m["role"] == "user" else "🤖"):
                st.markdown(m["content"])

        question = st.chat_input("Спросите что-нибудь по документу...")
        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user", avatar="🧑‍🎓"):
                st.markdown(question)

            client = get_client()
            system_blocks = [
                {"type": "text", "text": base_system(
                    "Отвечай точно и по существу. Если ответа в материале нет — честно скажи об этом."
                )},
                {
                    "type": "text",
                    "text": f"<document>\n{doc_text()}\n</document>",
                    "cache_control": {"type": "ephemeral"},
                },
            ]
            messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_history]

            with st.chat_message("assistant", avatar="🤖"):
                try:
                    def stream_answer():
                        with client.messages.stream(
                            model=model_id, max_tokens=2000, system=system_blocks, messages=messages
                        ) as stream:
                            for chunk in stream.text_stream:
                                yield chunk

                    answer = st.write_stream(stream_answer())
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except anthropic.AuthenticationError:
                    st.error("❌ Неверный API-ключ.")
                    st.session_state.chat_history.pop()
                except anthropic.RateLimitError:
                    st.error("⏳ Превышен лимит запросов. Попробуйте позже.")
                    st.session_state.chat_history.pop()
                except anthropic.APIError as e:
                    st.error(f"⚠️ Ошибка API: {e}")
                    st.session_state.chat_history.pop()
