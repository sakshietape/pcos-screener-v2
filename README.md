# PCOS Awareness Screener — v2

Everything from v1, plus: response storage + admin view, a free-text "Ask"
feature (Gemini), two new questions (family history, blood glucose flag),
a PCOS-familiarity gate with an optional brief, name capture, and English/Marathi
support.

## Files
- `app.py` — FastAPI backend (all routes below)
- `train_model.py` — trains `pcos_model.joblib` (unchanged from v1 — see note in "About the two new questions")
- `pcos_model.joblib` — trained model
- `pcos_adolescent_train_dataset.csv` — training data
- `index.html` — the chatbot frontend
- `requirements.txt`
- `.gitignore`
- **`doctor_avatar.jpg`** — copy this in from your v1 project folder; it isn't regenerated here

## Routes
| Route | Method | What it does |
|---|---|---|
| `/disclaimer?lang=en\|mr` | GET | Returns the opening disclaimer |
| `/pcos-info?lang=en\|mr` | GET | Short PCOS explainer, shown if user says they don't know what PCOS is |
| `/screen` | POST | Scores the answers, stores the response, returns band + message + why |
| `/ask` | POST | Free-text question → Gemini answer (needs `GEMINI_API_KEY`) |
| `/admin` | GET | Password-protected table of every stored response |

---

## 1. Run it locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```
(Windows: `python -m uvicorn app:app --reload` if `uvicorn` isn't found directly.)

Open `index.html` in your browser directly (double-click it). Make sure
`API_BASE` near the top of `index.html`'s `<script>` points at
`http://127.0.0.1:8000` for local testing.

---

## 2. Setting up the Ask feature (Gemini API)

1. Go to **aistudio.google.com** → sign in with Google → **Get API key** →
   create one. No credit card required. Free tier: ~1,500 requests/day.
2. **Locally:** set it as an environment variable before running the server:
   - Mac/Linux: `export GEMINI_API_KEY="your-key-here"`
   - Windows (PowerShell): `$env:GEMINI_API_KEY="your-key-here"`
   Then run `uvicorn app:app --reload` in that same terminal.
3. **On Render:** your service → **Environment** tab → **Add Environment
   Variable** → key `GEMINI_API_KEY`, value = your key → Save. Render
   redeploys automatically.
4. If the key isn't set, `/ask` still responds — with a clean error message
   instead of crashing — so the rest of the app keeps working either way.

**Never** put the key inside `index.html` or any frontend file — it must only
ever live on the backend/server side.

---

## 3. Viewing stored responses (the database)

Every completed screening is saved to a local SQLite file, `responses.db`,
created automatically the first time the server runs.

**To view them:** open `http://127.0.0.1:8000/admin` (or your Render URL +
`/admin`) in a browser. You'll be prompted for a username/password.

**Default credentials:** `admin` / `changeme` — **change these before you
deploy**, by setting environment variables `ADMIN_USER` and `ADMIN_PASS`
(same way as `GEMINI_API_KEY` above, both locally and on Render).

**Important limitation — read this before relying on it:** Render's **free**
web service tier does **not guarantee persistent disk** — data can be wiped
when the service restarts, redeploys, or sleeps and wakes up. This is fine
for a class demo (the data collected during your demo session will still be
there when you check `/admin` right after), but it is **not reliable for
long-term data collection**. If you need responses to survive indefinitely:
- Simplest free upgrade: create a free **Supabase** project (free Postgres
  database, persists forever on free tier) and swap the `sqlite3` calls in
  `app.py` for a Postgres connection (e.g. via `psycopg2` or `sqlalchemy`).
  Ask me when you're ready to do this and I'll write that version.
- Or: periodically download `responses.db` from Render's shell/logs before
  it might reset (manual, not recommended long-term).

---

## 4. About the two new questions (family history, blood glucose)

These two are **not** part of the trained model itself — your original
400-row dataset doesn't have columns for them, so there's no labeled data to
train on. Rather than fabricate training data, `app.py` applies a small,
clearly-documented **rule-based adjustment on top of** the trained model's
own probability: if either is answered "yes," a modest, capped boost is
added before the final low/some-indicators decision. This is intentional and
defensible — say exactly this if asked in a viva ("the base score comes from
a trained model; family history and glucose flags are a transparent
post-processing rule layered on top, since no training data exists for
them"). It's called out in a code comment in `app.py` too, right above where
it happens.

If you want a fully retrained model that includes these as real features,
you'd need new dataset rows with those two columns filled in and re-run
`train_model.py` with them added to the `features` list — happy to help with
that later if you can generate/simulate that data.

---

## 5. Deploying (same as v1)

1. Push this whole folder to GitHub (replace your old repo contents, or
   make a new repo — your call).
2. Render → your Web Service → it'll redeploy automatically on push, or
   trigger manually with "Manual Deploy."
3. Set `GEMINI_API_KEY`, `ADMIN_USER`, `ADMIN_PASS` under Environment (see
   above).
4. Update `API_BASE` in `index.html` to your live Render backend URL, push
   again.
5. Static site (frontend) auto-redeploys the same way.

---

## 6. Giving a colleague access

**GitHub (free, works exactly as you'd expect):**
1. Your repo → **Settings** → **Collaborators** → **Add people**.
2. Enter their GitHub username or email → they get an invite → once
   accepted, they can push/pull like you can.

**Render (important — this changed in April 2026):**
Render's free "Hobby" workspace is now limited to **one member only** —
you can't invite a collaborator to your workspace for free anymore; that
requires the paid Pro plan (~$25/month).

**Free workaround that works just as well for a student project:**
- Your colleague creates their **own free Render account**.
- They connect **their own account** to the **same GitHub repo** (once
  you've added them as a GitHub collaborator, they can see and deploy from
  it).
- They create their own Web Service + Static Site pointing at that repo —
  effectively their own independent deployment of the same code.
- You don't share passwords or a login; you both just work off the same
  GitHub source, each with your own free Render deployment.

This is actually a cleaner setup than shared workspace access for a
two-person student project — no shared credentials, no risk of one person's
changes affecting the other's live demo unexpectedly.

---

## 7. Defense/Viva quick answers (updated)

- **"Where are responses stored?"** — A local SQLite database on the
  backend, viewable at `/admin` behind a username/password.
- **"How does the chatbot understand free-text questions?"** — The `/ask`
  endpoint sends the question to Google's Gemini model with a strict system
  prompt limiting it to non-diagnostic PCOS/health-literacy topics, in
  the user's selected language.
- **"Is the whole model bilingual?"** — The scripted question flow and all
  UI text is available in English and Marathi; the free-text Ask feature
  also responds in whichever language is selected.
- **"Why aren't family history and blood glucose part of the trained
  model?"** — see Section 4 above.
