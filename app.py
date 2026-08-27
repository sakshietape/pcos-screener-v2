import os
import sqlite3
import json
from datetime import datetime, timezone
from contextlib import contextmanager

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
import joblib
import numpy as np
import secrets

# ─────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────
bundle = joblib.load("pcos_model.joblib")
model = bundle["model"]
FEATURES = bundle["features"]
THRESHOLD = bundle["threshold"]

DB_PATH = "responses.db"

app = FastAPI(title="PCOS Awareness Screener API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────
# Database — Postgres (permanent) when DATABASE_URL is set,
# otherwise falls back to a local SQLite file for quick local testing.
#
# Render's free-tier disk does NOT persist SQLite across restarts —
# that's why responses were disappearing. Set DATABASE_URL to a free
# Supabase Postgres connection string (see README) to store responses
# permanently. Everything below auto-detects which one to use.
# ─────────────────────────────────────────────────────────────────

DATABASE_URL = os.environ.get("DATABASE_URL")
USE_POSTGRES = bool(DATABASE_URL)

if USE_POSTGRES:
    import psycopg2
    import psycopg2.extras


@contextmanager
def get_db():
    if USE_POSTGRES:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        cur = conn.cursor() if USE_POSTGRES else conn
        id_col = "id SERIAL PRIMARY KEY" if USE_POSTGRES else "id INTEGER PRIMARY KEY AUTOINCREMENT"
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS responses (
                {id_col},
                timestamp TEXT NOT NULL,
                name TEXT,
                language TEXT,
                age INTEGER,
                years_since_menarche INTEGER,
                bmi REAL,
                acne INTEGER,
                irregular_periods INTEGER,
                facial_hair_growth INTEGER,
                weight_gain INTEGER,
                bloating INTEGER,
                hair_thinning INTEGER,
                fatigue INTEGER,
                family_history INTEGER,
                blood_glucose_flag INTEGER,
                probability REAL,
                band TEXT
            )
        """)

init_db()

# ─────────────────────────────────────────────────────────────────
# Static copy (English + Marathi) — used by /disclaimer and /pcos-info
# The frontend also keeps its own copies for the question flow;
# these two endpoints are the "server-confirmed" source of truth.
# ─────────────────────────────────────────────────────────────────

TEXT = {
    "en": {
        "disclaimer": (
            "This tool looks at common symptoms to help you understand your body "
            "better — it does not diagnose PCOS or any medical condition. Many of "
            "these symptoms (like irregular periods) are completely normal for "
            "teenagers, especially in the first couple of years after your first "
            "period. Only a doctor can tell you what's really going on, using "
            "proper tests."
        ),
        "pcos_info": (
            "PCOS (Polycystic Ovary Syndrome) is a common hormonal condition. It "
            "can cause irregular periods, acne, extra hair growth, and weight "
            "changes. It affects many teenagers and adults, and it's very "
            "manageable once identified — a doctor can confirm it with simple "
            "tests. Having some of these symptoms doesn't mean you have PCOS; "
            "lots of things can cause them."
        ),
        "low_indicators": (
            "Your responses don't show many common signs linked to PCOS right "
            "now. If anything changes or you're worried, it's always okay to "
            "talk to a doctor or trusted adult."
        ),
        "some_indicators": (
            "A few of your responses overlap with symptoms sometimes seen in "
            "PCOS. This is common and doesn't mean you have it — many things can "
            "cause these symptoms. It would be a good idea to mention this to a "
            "parent/guardian and a doctor, just so they can check properly."
        ),
    },
    "mr": {
        "disclaimer": (
            "हे साधन तुमच्या शरीराबद्दल अधिक समजून घेण्यासाठी सामान्य लक्षणांकडे पाहते — "
            "हे PCOS किंवा कोणत्याही वैद्यकीय स्थितीचे निदान करत नाही. यातील अनेक लक्षणे "
            "(जसे की अनियमित मासिक पाळी) किशोरवयीन मुलींसाठी पूर्णपणे सामान्य असू शकतात, "
            "विशेषतः पहिल्या मासिक पाळीनंतरच्या पहिल्या एक-दोन वर्षांत. योग्य तपासण्या करून "
            "फक्त डॉक्टरच खरी परिस्थिती सांगू शकतात."
        ),
        "pcos_info": (
            "PCOS (पॉलिसिस्टिक ओव्हरी सिंड्रोम) ही एक सामान्य हार्मोनल स्थिती आहे. यामुळे "
            "अनियमित मासिक पाळी, मुरुम, जास्त केस वाढणे आणि वजनातील बदल होऊ शकतात. ही "
            "स्थिती अनेक किशोरवयीन मुली आणि प्रौढांमध्ये आढळते आणि ओळखल्यानंतर ती सहज "
            "व्यवस्थापित करता येते — डॉक्टर साध्या तपासण्यांद्वारे याची खात्री करू शकतात. "
            "यातील काही लक्षणे असणे म्हणजे PCOS असणे असे नाही; इतर अनेक कारणांमुळेही ही "
            "लक्षणे दिसू शकतात."
        ),
        "low_indicators": (
            "तुमच्या उत्तरांमध्ये सध्या PCOS शी संबंधित फारशी सामान्य लक्षणे दिसत नाहीत. "
            "काही बदलल्यास किंवा काळजी वाटल्यास डॉक्टर किंवा विश्वासू मोठ्या व्यक्तीशी "
            "बोलणे केव्हाही योग्य आहे."
        ),
        "some_indicators": (
            "तुमच्या काही उत्तरांमध्ये PCOS मध्ये कधीकधी दिसणाऱ्या लक्षणांशी साम्य आहे. "
            "हे सामान्य आहे आणि याचा अर्थ तुम्हाला PCOS आहे असे नाही — या लक्षणांची अनेक "
            "कारणे असू शकतात. हे पालक/पालक आणि डॉक्टरांना सांगणे चांगली कल्पना ठरेल, "
            "जेणेकरून ते व्यवस्थित तपासणी करू शकतील."
        ),
    },
}


@app.get("/disclaimer")
def get_disclaimer(lang: str = "en"):
    lang = lang if lang in TEXT else "en"
    return {"disclaimer": TEXT[lang]["disclaimer"]}


@app.get("/pcos-info")
def get_pcos_info(lang: str = "en"):
    lang = lang if lang in TEXT else "en"
    return {"info": TEXT[lang]["pcos_info"]}


# ─────────────────────────────────────────────────────────────────
# /screen — scoring endpoint (now with 2 extra questions + storage)
# ─────────────────────────────────────────────────────────────────

class ScreeningInput(BaseModel):
    name: str = "Friend"
    language: str = "en"
    age: int
    years_since_menarche: int
    bmi: float
    acne: bool
    irregular_periods: bool
    facial_hair_growth: bool
    weight_gain: bool
    bloating: bool
    hair_thinning: bool
    fatigue: bool
    family_history: bool = False
    blood_glucose_flag: bool = False


def explain(data: ScreeningInput, irregular_counts: bool) -> str:
    reasons = []
    if irregular_counts:
        reasons.append("irregular periods")
    if data.facial_hair_growth:
        reasons.append("increased facial hair")
    if data.hair_thinning:
        reasons.append("hair thinning")
    if data.weight_gain:
        reasons.append("weight changes")
    if data.acne:
        reasons.append("acne")
    if data.bloating:
        reasons.append("bloating")
    if data.fatigue:
        reasons.append("fatigue")
    if data.family_history:
        reasons.append("family history of PCOS")
    if data.blood_glucose_flag:
        reasons.append("blood sugar concerns mentioned by a doctor")
    return ", ".join(reasons) if reasons else "no strongly weighted symptoms"


@app.post("/screen")
def screen(data: ScreeningInput):
    irregular_counts = data.irregular_periods and data.years_since_menarche >= 2

    X = np.array([[
        data.age,
        data.years_since_menarche,
        data.bmi,
        int(data.acne),
        int(irregular_counts),
        int(data.facial_hair_growth),
        int(data.weight_gain),
        int(data.bloating),
        int(data.hair_thinning),
        int(data.fatigue),
    ]])

    prob = float(model.predict_proba(X)[0][1])

    # ── Documented heuristic layer ──────────────────────────────
    # family_history and blood_glucose_flag are NOT in the original
    # training data (no labeled rows exist for them), so they are
    # not part of the trained model itself. Instead we apply a small,
    # clinically-justified adjustment on top of the model's own
    # probability: both family history of PCOS and clinician-flagged
    # blood sugar concerns are well-established real-world risk
    # correlates. This is a transparent rule, not a learned weight —
    # say so explicitly if asked in a defense/viva.
    if data.family_history:
        prob = min(prob + 0.05, 0.97)
    if data.blood_glucose_flag:
        prob = min(prob + 0.05, 0.97)

    lang = data.language if data.language in TEXT else "en"
    band = "some_indicators" if prob >= THRESHOLD else "low_indicators"
    message = TEXT[lang][band]

    # personalize with name
    if data.name and data.name.strip() and data.name.strip().lower() != "friend":
        message = f"{data.name.strip()}, " + message[0].lower() + message[1:]

    # ── Store in DB (never expose raw probability to the client) ──
    ph = "%s" if USE_POSTGRES else "?"
    insert_sql = f"""
        INSERT INTO responses (
            timestamp, name, language, age, years_since_menarche, bmi,
            acne, irregular_periods, facial_hair_growth, weight_gain,
            bloating, hair_thinning, fatigue, family_history,
            blood_glucose_flag, probability, band
        ) VALUES ({",".join([ph] * 17)})
    """
    insert_values = (
        datetime.now(timezone.utc).isoformat(),
        data.name.strip() if data.name else "",
        lang, data.age, data.years_since_menarche, data.bmi,
        int(data.acne), int(data.irregular_periods), int(data.facial_hair_growth),
        int(data.weight_gain), int(data.bloating), int(data.hair_thinning),
        int(data.fatigue), int(data.family_history), int(data.blood_glucose_flag),
        round(prob, 4), band,
    )
    with get_db() as conn:
        if USE_POSTGRES:
            cur = conn.cursor()
            cur.execute(insert_sql, insert_values)
        else:
            conn.execute(insert_sql, insert_values)

    return {
        "band": band,
        "message": message,
        "why": explain(data, irregular_counts),
    }


# ─────────────────────────────────────────────────────────────────
# /admin — password-protected view of stored responses
# ─────────────────────────────────────────────────────────────────

security = HTTPBasic()
ADMIN_USER = os.environ.get("ADMIN_USER", "admin")
ADMIN_PASS = os.environ.get("ADMIN_PASS", "changeme")


def check_admin(credentials: HTTPBasicCredentials = Depends(security)):
    correct_user = secrets.compare_digest(credentials.username, ADMIN_USER)
    correct_pass = secrets.compare_digest(credentials.password, ADMIN_PASS)
    if not (correct_user and correct_pass):
        raise HTTPException(status_code=401, detail="Invalid credentials", headers={"WWW-Authenticate": "Basic"})
    return True


@app.get("/admin", response_class=HTMLResponse)
def admin_view(_: bool = Depends(check_admin)):
    with get_db() as conn:
        if USE_POSTGRES:
            cur = conn.cursor()
            cur.execute("SELECT * FROM responses ORDER BY id DESC")
            rows = cur.fetchall()
        else:
            rows = conn.execute("SELECT * FROM responses ORDER BY id DESC").fetchall()

    table_rows = ""
    for r in rows:
        table_rows += f"""
        <tr>
            <td>{r['id']}</td>
            <td>{r['timestamp'][:19].replace('T',' ')}</td>
            <td>{r['name']}</td>
            <td>{r['language']}</td>
            <td>{r['age']}</td>
            <td>{r['years_since_menarche']}</td>
            <td>{r['bmi']}</td>
            <td>{'Y' if r['acne'] else '-'}</td>
            <td>{'Y' if r['irregular_periods'] else '-'}</td>
            <td>{'Y' if r['facial_hair_growth'] else '-'}</td>
            <td>{'Y' if r['weight_gain'] else '-'}</td>
            <td>{'Y' if r['bloating'] else '-'}</td>
            <td>{'Y' if r['hair_thinning'] else '-'}</td>
            <td>{'Y' if r['fatigue'] else '-'}</td>
            <td>{'Y' if r['family_history'] else '-'}</td>
            <td>{'Y' if r['blood_glucose_flag'] else '-'}</td>
            <td>{r['probability']}</td>
            <td><strong>{r['band']}</strong></td>
        </tr>"""

    html = f"""
    <html>
    <head>
        <title>PCOS Screener — Admin</title>
        <style>
            body {{ font-family: -apple-system, sans-serif; background:#f5f3ff; margin:0; padding:24px; }}
            h1 {{ color:#4C1D95; }}
            table {{ border-collapse: collapse; width:100%; background:white; border-radius:8px; overflow:hidden; box-shadow:0 2px 10px rgba(0,0,0,.06); }}
            th, td {{ padding:8px 10px; text-align:left; font-size:12.5px; border-bottom:1px solid #eee; }}
            th {{ background:#6D28D9; color:white; position:sticky; top:0; }}
            tr:hover {{ background:#f8f7ff; }}
            .count {{ color:#6b6480; margin-bottom:12px; }}
        </style>
    </head>
    <body>
        <h1>PCOS Screener — Response Log</h1>
        <p class="count">{len(rows)} responses stored</p>
        <table>
            <tr>
                <th>ID</th><th>Timestamp (UTC)</th><th>Name</th><th>Lang</th><th>Age</th>
                <th>Yrs since menarche</th><th>BMI</th><th>Acne</th><th>Irregular periods</th>
                <th>Facial hair</th><th>Weight gain</th><th>Bloating</th><th>Hair thinning</th>
                <th>Fatigue</th><th>Family history</th><th>Blood glucose flag</th>
                <th>Probability</th><th>Band</th>
            </tr>
            {table_rows}
        </table>
    </body>
    </html>
    """
    return HTMLResponse(content=html)
