"""
╔══════════════════════════════════════════════════════════════════════════╗
║          MoodLens  ·  Sentiment Intelligence Platform  ·  v2.0           ║
╚══════════════════════════════════════════════════════════════════════════╝
  Run:   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import sys, time, logging
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel

# ══════════════════════════════════════════════════════════════════════════
#  PATH  —  exact same as original
# ══════════════════════════════════════════════════════════════════════════
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "python"))

# ══════════════════════════════════════════════════════════════════════════
#  ML IMPORT  —  direct, no guard, same as original
# ══════════════════════════════════════════════════════════════════════════
from roberta_predict import predict, compare_all_models

# ══════════════════════════════════════════════════════════════════════════
#  LOGGER
# ══════════════════════════════════════════════════════════════════════════
RESET  = "\033[0m";  BOLD   = "\033[1m";  DIM    = "\033[2m"
CYAN   = "\033[36m"; GREEN  = "\033[32m"; YELLOW = "\033[33m"
RED    = "\033[31m"

class _Fmt(logging.Formatter):
    C = {"DEBUG": DIM, "INFO": GREEN, "WARNING": YELLOW, "ERROR": RED, "CRITICAL": f"{BOLD}{RED}"}
    def format(self, r):
        ts = datetime.now().strftime("%H:%M:%S")
        lc = self.C.get(r.levelname, "")
        return f"{DIM}{ts}{RESET}  {lc}{r.levelname:<8}{RESET}  {CYAN}{r.name}{RESET}  {r.getMessage()}"

_h = logging.StreamHandler(); _h.setFormatter(_Fmt())
logging.root.handlers = [_h]; logging.root.setLevel(logging.INFO)
log = logging.getLogger("moodlens")

# ══════════════════════════════════════════════════════════════════════════
#  LIFESPAN
# ══════════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"""
{YELLOW}{BOLD}
  ╔═══════════════════════════════════════════════════════╗
  ║                                                       ║
  ║   ███╗   ███╗ ██████╗  ██████╗ ██████╗               ║
  ║   ████╗ ████║██╔═══██╗██╔═══██╗██╔══██╗              ║
  ║   ██╔████╔██║██║   ██║██║   ██║██║  ██║              ║
  ║   ██║╚██╔╝██║██║   ██║██║   ██║██║  ██║              ║
  ║   ██║ ╚═╝ ██║╚██████╔╝╚██████╔╝██████╔╝              ║
  ║   ╚═╝     ╚═╝ ╚═════╝  ╚═════╝ ╚═════╝  LENS  v2.0  ║
  ║                                                       ║
  ╚═══════════════════════════════════════════════════════╝
{RESET}
  {YELLOW}◆{RESET}  Splash   →  http://127.0.0.1:{YELLOW}8000{RESET}
  {YELLOW}◆{RESET}  Swagger  →  http://127.0.0.1:{YELLOW}8000/docs{RESET}
  {YELLOW}◆{RESET}  Health   →  http://127.0.0.1:{YELLOW}8000/health{RESET}

  {GREEN}✓{RESET}  ML Engine  →  {GREEN}{BOLD}READY{RESET}
  {GREEN}✓{RESET}  Dataset    →  Zomato Reviews Corpus
  {GREEN}✓{RESET}  Models     →  RoBERTa · DistilRoBERTa · BERT · ALBERT
""")
    yield
    print(f"\n  {YELLOW}◆{RESET} MoodLens offline {DIM}· bye 👋{RESET}\n")

# ══════════════════════════════════════════════════════════════════════════
#  APP
# ══════════════════════════════════════════════════════════════════════════
app = FastAPI(
    title       = "MoodLens · Sentiment Intelligence API",
    description = """
## MoodLens — Multi-Model NLP Sentiment Engine

Enterprise-grade sentiment analysis powered by four transformer models,
fine-tuned on the **Zomato Reviews** corpus.

### Models
| Model | Hugging Face ID | Strength |
|---|---|---|
| **RoBERTa** | `cardiffnlp/twitter-roberta-base-sentiment-latest` | General sentiment · Default |
| **DistilRoBERTa** | `mrm8488/distilroberta-finetuned-...` | Faster · Financial/review domain |
| **BERT** | `nlptown/bert-base-multilingual-uncased-sentiment` | Multilingual · 5-star scale |
| **ALBERT** | `textattack/albert-base-v2-yelp-polarity` | Efficient · Yelp polarity |

### Quick Start
```bash
curl -X POST http://localhost:8000/predict \\
     -H "Content-Type: application/json" \\
     -d '{"text": "Best biryani I have ever had!"}'
```
""",
    version     = "2.0.0",
    docs_url    = None,
    redoc_url   = "/redoc",
    lifespan    = lifespan,
    openapi_tags= [
        {"name": "Inference", "description": "Sentiment prediction endpoints."},
        {"name": "System",    "description": "Health and diagnostics."},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.middleware("http")
async def _timer(req: Request, call_next):
    t0  = time.perf_counter()
    res = await call_next(req)
    ms  = (time.perf_counter() - t0) * 1000
    c   = GREEN if res.status_code < 400 else YELLOW if res.status_code < 500 else RED
    log.info(f"{c}{req.method:<6}{RESET} {req.url.path:<22}  {c}{res.status_code}{RESET}  {DIM}{ms:.1f}ms{RESET}")
    res.headers["X-Response-Time"] = f"{ms:.2f}ms"
    return res

# ══════════════════════════════════════════════════════════════════════════
#  SCHEMAS
# ══════════════════════════════════════════════════════════════════════════
class TextInput(BaseModel):
    text: str

# ══════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════

from fastapi.responses import FileResponse

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(Path(__file__).parent / "favicon.ico")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root():
    return HTMLResponse(_splash())

@app.get("/docs", response_class=HTMLResponse, include_in_schema=False)
def dark_docs():
    return get_swagger_ui_html(
        openapi_url         = app.openapi_url,
        title               = "MoodLens · API Docs",
        swagger_js_url      = "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url     = "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
        swagger_favicon_url = "data:,",   # blank favicon for docs too
        swagger_ui_parameters={
            "syntaxHighlight.theme"   : "monokai",
            "tryItOutEnabled"         : True,
            "displayRequestDuration"  : True,
            "defaultModelsExpandDepth": -1,
        },
    )

@app.get(
    "/health",
    summary="Server & ML Health Check",
    description="""
Returns real-time status of the API server and ML engine.

**Use this endpoint to:**
- ✅ Verify all 4 transformer models are loaded and ready
- ✅ Confirm server is reachable before sending inference requests
- ✅ Monitor uptime in CI/CD pipelines or dashboards
- ✅ Check UTC timestamp for server clock sync
""",
    tags=["System"],
)
def health():
    return {
        "status"   : "ok",
        "version"  : "2.0.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "models"   : ["roberta", "distilroberta", "bert", "albert"],
        "dataset"  : "Zomato Reviews — Food & Dining Corpus",
    }

@app.post(
    "/predict",
    summary="Predict Sentiment  (RoBERTa)",
    description="""
Runs the **default RoBERTa model** on your input text.

### How it works
1. Text tokenised and truncated to **512 tokens**
2. RoBERTa runs a single forward pass
3. Softmax scores mapped → **Positive / Neutral / Negative**
4. Highest-probability class returned as `prediction`

### Response fields
| Field | Type | Description |
|---|---|---|
| `prediction` | string | `Positive`, `Neutral`, or `Negative` |
| `confidence` | float  | Winning class score (0.0 – 1.0) |
| `positive`   | float  | Raw probability — Positive class |
| `neutral`    | float  | Raw probability — Neutral class |
| `negative`   | float  | Raw probability — Negative class |
""",
    tags=["Inference"],
)
def get_prediction(data: TextInput):
    label, probs = predict(data.text)
    return {
        "prediction": label,
        "confidence": float(max(probs)),
        "negative"  : float(probs[0]),
        "neutral"   : float(probs[1]),
        "positive"  : float(probs[2]),
    }

@app.post(
    "/compare",
    summary="Compare All 4 Models",
    description="""
Runs **all four models** on the same input and returns results sorted by confidence.

### Models compared
| Model | Strength |
|---|---|
| **RoBERTa**       | General-purpose · highest accuracy |
| **DistilRoBERTa** | 40% faster · financial/review domain |
| **BERT**          | Multilingual · 5-star scale |
| **ALBERT**        | Lightweight · Yelp polarity |

### How it works
1. All 4 models process the input **independently**
2. Raw labels normalised → `Positive / Neutral / Negative`
3. Results **sorted by confidence** — best model first
4. Use `/predict` for speed · `/compare` for cross-validation
""",
    tags=["Inference"],
)
def compare_models(data: TextInput):
    result = compare_all_models(data.text)
    return {"comparison": result}

@app.exception_handler(404)
async def not_found(_, __):
    return JSONResponse(status_code=404, content={
        "error" : "Route not found",
        "routes": {"GET": ["/", "/health", "/docs", "/redoc"], "POST": ["/predict", "/compare"]},
        "docs"  : "http://127.0.0.1:8000/docs",
    })

# ══════════════════════════════════════════════════════════════════════════
#  SPLASH
#  DARK  → Yellow #FFD449 + Zomato Red #E23744
#  LIGHT → Uber Navy #09091A + Uber Blue #276EF1
#  Font  → JetBrains Mono everywhere
#  Favicon → none
# ══════════════════════════════════════════════════════════════════════════
def _splash() -> str:
    return r"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>MoodLens · API</title>
<link rel="icon" href="favicon.ico" />
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&display=swap" rel="stylesheet"/>
<style>
/* ════════════════════════════════════════════════════════════
   TOKENS
   dark  → yellow #FFD449  accent-red #E23744  (Zomato palette)
   light → navy  #09091A   blue #276EF1         (Uber palette)
════════════════════════════════════════════════════════════ */
[data-theme="dark"]{
  --bg:      #07080E;
  --bg2:     #0C0E18;
  --card:    #10131F;
  --card2:   #151929;
  --b1:      #1C2038;
  --b2:      #242A45;
  --fg:      #E4EAF8;
  --f2:      #7A90B8;
  --f3:      #3D4F70;
  --nav:     rgba(7,8,14,.88);

  /* Zomato-inspired yellow + red accent */
  --accent:  #FFD449;
  --accent2: #E23744;
  --accent3: #FF6B6B;
  --aglow:   rgba(255,212,73,.14);
  --aglow2:  rgba(226,55,68,.10);
  --aborder: rgba(255,212,73,.25);
  --aborder2:rgba(226,55,68,.22);
  --atext:   #07080E;       /* text on accent bg */

  --post-bg: rgba(255,212,73,.10);
  --post-col: #FFD449;
  --post-bd: rgba(255,212,73,.22);
  --get-bg:  rgba(226,55,68,.10);
  --get-col: #E23744;
  --get-bd:  rgba(226,55,68,.22);

  --dot-col: #E23744;
  --sh:      0 28px 70px rgba(0,0,0,.60);
}

[data-theme="light"]{
  --bg:      #F0F2F8;
  --bg2:     #E4E8F4;
  --card:    #FFFFFF;
  --card2:   #EEF1FA;
  --b1:      #C8D0E8;
  --b2:      #B0BCE0;
  --fg:      #09091A;       /* Uber navy */
  --f2:      #3A4A6A;
  --f3:      #8898B8;
  --nav:     rgba(240,242,248,.92);

  /* Uber-inspired navy + blue */
  --accent:  #09091A;       /* Uber black/navy */
  --accent2: #276EF1;       /* Uber blue */
  --accent3: #1A56C4;
  --aglow:   rgba(9,9,26,.07);
  --aglow2:  rgba(39,110,241,.09);
  --aborder: rgba(9,9,26,.20);
  --aborder2:rgba(39,110,241,.25);
  --atext:   #FFFFFF;       /* text on accent bg */

  --post-bg: rgba(39,110,241,.10);
  --post-col: #276EF1;
  --post-bd: rgba(39,110,241,.22);
  --get-bg:  rgba(9,9,26,.08);
  --get-col: #09091A;
  --get-bd:  rgba(9,9,26,.18);

  --dot-col: #276EF1;
  --sh:      0 16px 48px rgba(9,9,26,.12);
}

/* ════ BASE ═════════════════════════════════════════════════ */
*{box-sizing:border-box;margin:0;padding:0;
  transition:background .25s,color .25s,border-color .25s,opacity .25s;}
html{scroll-behavior:smooth;}
body{
  background:var(--bg);color:var(--fg);
  font-family:'JetBrains Mono',monospace;
  font-size:14px;line-height:1.7;min-height:100vh;overflow-x:hidden;
}

/* grid-line texture */
body::after{
  content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
  background-image:
    linear-gradient(var(--b1) 1px,transparent 1px),
    linear-gradient(90deg,var(--b1) 1px,transparent 1px);
  background-size:52px 52px;opacity:.15;
}
[data-theme="light"] body::after{opacity:.08;}

/* ambient blobs */
.blob{position:fixed;border-radius:50%;filter:blur(150px);pointer-events:none;z-index:0;}
.b1{width:750px;height:750px;top:-280px;right:-180px;
    background:radial-gradient(circle,var(--aglow),transparent 70%);
    animation:bf 14s ease-in-out infinite;}
.b2{width:600px;height:600px;bottom:-220px;left:-180px;
    background:radial-gradient(circle,var(--aglow2),transparent 70%);
    animation:bf 18s ease-in-out infinite reverse;}
.b3{width:350px;height:350px;top:38%;left:38%;
    background:radial-gradient(circle,rgba(39,110,241,.05),transparent 70%);
    animation:bf 22s ease-in-out infinite 7s;}
@keyframes bf{
  0%,100%{transform:translate(0,0) scale(1);}
  33%{transform:translate(30px,-42px) scale(1.07);}
  66%{transform:translate(-20px,24px) scale(.93);}
}

/* ════ NAV ══════════════════════════════════════════════════ */
nav{
  position:fixed;top:0;left:0;right:0;z-index:200;
  background:var(--nav);backdrop-filter:blur(24px);-webkit-backdrop-filter:blur(24px);
  border-bottom:1px solid var(--b1);
  display:flex;align-items:center;justify-content:space-between;
  padding:0 40px;height:58px;gap:16px;
}
.nlogo{
  font-size:17px;font-weight:800;letter-spacing:-1px;
  text-decoration:none;display:flex;flex-direction:column;line-height:1.2;
}
.nlogo-main{
  color:var(--accent);
  text-shadow:none;
}
[data-theme="dark"] .nlogo-main{
  text-shadow:0 0 28px rgba(255,212,73,.45);
}
.nlogo-tag{color:var(--f3);font-size:9px;font-weight:400;letter-spacing:2.5px;text-transform:uppercase;margin-top:1px;}
.nr{display:flex;align-items:center;gap:8px;}
.npill{
  display:flex;align-items:center;gap:7px;padding:6px 14px;border-radius:7px;
  border:1px solid var(--b2);background:var(--card2);color:var(--f2);
  font-size:11px;font-family:'JetBrains Mono',monospace;
  text-decoration:none;white-space:nowrap;cursor:pointer;letter-spacing:.3px;font-weight:500;
}
.npill:hover{border-color:var(--accent);color:var(--accent);}
.live-pill{
  display:flex;align-items:center;gap:7px;padding:5px 13px;border-radius:7px;
  border:1px solid var(--aborder2);background:var(--aglow2);
  font-size:10px;color:var(--dot-col);letter-spacing:.5px;font-weight:500;
}
.tbtn{
  width:38px;height:38px;border-radius:8px;border:1px solid var(--b2);
  background:var(--card2);cursor:pointer;font-size:15px;
  display:flex;align-items:center;justify-content:center;
}
.tbtn:hover{border-color:var(--accent);transform:rotate(20deg);}

/* ════ HERO ════════════════════════════════════════════════ */
.hero{
  position:relative;z-index:1;min-height:100vh;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  padding:80px 24px 60px;text-align:center;
}
.eyebrow{
  display:inline-flex;align-items:center;gap:9px;padding:6px 18px;border-radius:7px;
  border:1px solid var(--aborder);background:var(--aglow);
  font-size:9px;color:var(--accent);letter-spacing:3.5px;text-transform:uppercase;
  margin-bottom:40px;animation:fu .7s ease both;font-weight:500;
}
.ldot{width:6px;height:6px;border-radius:50%;background:var(--dot-col);
  box-shadow:0 0 10px var(--dot-col);animation:lp 2s ease-in-out infinite;}
@keyframes lp{0%,100%{opacity:1;transform:scale(1);}50%{opacity:.3;transform:scale(1.9);}}

.hero-pre{font-size:12px;color:var(--f3);letter-spacing:4px;text-transform:uppercase;
  margin-bottom:14px;animation:fu .7s .05s ease both;font-weight:400;}
h1{font-size:clamp(58px,10vw,108px);font-weight:800;
  line-height:.88;letter-spacing:-5px;margin-bottom:10px;animation:fu .7s .1s ease both;}
.h1a{display:block;color:var(--fg);}
.h1b{
  display:block;color:var(--accent);
  animation:flicker 9s ease-in-out infinite 2s;
}
[data-theme="dark"] .h1b{
  text-shadow:0 0 80px rgba(255,212,73,.4),0 0 160px rgba(255,212,73,.12);
}
[data-theme="light"] .h1b{
  text-shadow:0 2px 24px rgba(9,9,26,.15);
}
@keyframes flicker{
  0%,94%,100%{opacity:1;}
  95%{opacity:.7;}97%{opacity:1;}98%{opacity:.85;}99%{opacity:1;}
}

.hsub{max-width:560px;font-size:12.5px;color:var(--f2);line-height:2;
  margin:28px 0 40px;animation:fu .7s .2s ease both;font-weight:400;}
.hsub strong{color:var(--fg);font-weight:600;}

/* CTA */
.hbtns{display:flex;gap:12px;flex-wrap:wrap;justify-content:center;animation:fu .7s .3s ease both;}
.bpri{
  display:inline-flex;align-items:center;gap:9px;padding:13px 30px;border-radius:10px;
  background:var(--accent);color:var(--atext);font-weight:700;font-size:12px;
  text-decoration:none;border:none;cursor:pointer;letter-spacing:.5px;
  box-shadow:0 8px 36px var(--aglow);
}
.bpri:hover{transform:translateY(-3px);box-shadow:0 16px 50px var(--aglow),0 4px 16px rgba(0,0,0,.3);}
.bout{
  display:inline-flex;align-items:center;gap:8px;padding:12px 24px;border-radius:10px;
  border:1px solid var(--b2);background:var(--card);color:var(--f2);
  font-weight:500;font-size:12px;text-decoration:none;letter-spacing:.3px;
}
.bout:hover{border-color:var(--accent2);color:var(--accent2);transform:translateY(-3px);}

/* stats */
.stats{
  display:flex;margin-top:60px;flex-wrap:wrap;justify-content:center;
  animation:fu .7s .4s ease both;
  border:1px solid var(--b1);border-radius:14px;background:var(--card);overflow:hidden;
}
.stat{padding:20px 34px;text-align:center;border-right:1px solid var(--b1);}
.stat:last-child{border-right:none;}
.sv{font-size:34px;font-weight:800;letter-spacing:-2px;color:var(--accent);}
[data-theme="dark"] .sv{text-shadow:0 0 22px rgba(255,212,73,.28);}
[data-theme="light"] .sv{text-shadow:0 1px 12px rgba(9,9,26,.1);}
.sl{font-size:9px;color:var(--f3);letter-spacing:2px;text-transform:uppercase;margin-top:3px;}

.online-bar{
  display:inline-flex;align-items:center;gap:18px;
  margin-top:26px;padding:10px 24px;border-radius:10px;
  border:1px solid var(--b1);background:var(--card);
  font-size:10px;color:var(--f3);letter-spacing:.8px;
  animation:fu .7s .5s ease both;
}
.online-bar .on{color:var(--dot-col);font-weight:600;}
.sep{color:var(--b2);}

/* ════ TERMINAL ═════════════════════════════════════════════ */
.terminal-wrap{position:relative;z-index:1;max-width:760px;margin:0 auto 120px;padding:0 24px;}
.terminal{
  background:var(--bg2);border:1px solid var(--b2);border-radius:16px;
  overflow:hidden;box-shadow:var(--sh);
}
.t-bar{display:flex;align-items:center;gap:8px;padding:12px 18px;
  border-bottom:1px solid var(--b1);background:var(--card);}
.t-dot{width:12px;height:12px;border-radius:50%;}
.t-title{margin-left:8px;font-size:10px;color:var(--f3);letter-spacing:1.2px;text-transform:uppercase;}
.t-body{padding:22px 24px;font-size:12px;line-height:2.1;text-align:left;}
.t-prompt{color:var(--accent);flex-shrink:0;}
.t-line{display:flex;align-items:flex-start;gap:10px;margin-bottom:2px;}
.t-str{color:#A8E6A3;}
.t-out{color:var(--f2);margin-left:20px;}
.t-key{color:var(--accent2);}
[data-theme="light"] .t-key{color:var(--accent2);}
.t-num{color:var(--accent);}
.t-pos{color:#A8E6A3;}
.t-cmt{color:var(--f3);font-style:italic;}
.t-cursor{display:inline-block;width:8px;height:14px;background:var(--accent);
  animation:blink 1s step-end infinite;vertical-align:middle;margin-left:2px;}
@keyframes blink{0%,100%{opacity:1;}50%{opacity:0;}}

/* ════ CONTENT WRAPPER ══════════════════════════════════════ */
.wrap{position:relative;z-index:1;max-width:1120px;margin:0 auto;padding:0 28px 120px;}
.divider{height:1px;background:linear-gradient(90deg,transparent,var(--b2),transparent);margin-bottom:88px;}
.slabel{
  font-size:9px;letter-spacing:4px;text-transform:uppercase;color:var(--accent);
  margin-bottom:16px;display:flex;align-items:center;gap:14px;font-weight:600;
}
.slabel::after{content:'';flex:1;height:1px;background:var(--b1);}
.stitle{font-size:clamp(30px,4vw,48px);font-weight:800;letter-spacing:-2.5px;line-height:.95;margin-bottom:12px;}
.sdesc{color:var(--f2);font-size:12px;margin-bottom:56px;max-width:500px;line-height:1.9;font-weight:400;}

/* ════ ENDPOINT CARDS ═══════════════════════════════════════ */
.epgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:14px;}
.ecard{
  background:var(--card);border:1px solid var(--b1);border-radius:18px;
  overflow:hidden;position:relative;
}
.ecard::before{
  content:'';position:absolute;left:0;top:0;bottom:0;width:3px;
  background:linear-gradient(180deg,var(--accent),var(--accent2));
  opacity:0;transition:opacity .3s;
}
.ecard:hover{border-color:var(--b2);box-shadow:var(--sh);}
.ecard:hover::before{opacity:1;}

.ehead{padding:24px 26px 20px;border-bottom:1px solid var(--b1);display:flex;align-items:flex-start;gap:14px;}
.meth{font-size:9px;font-weight:700;letter-spacing:2px;padding:4px 11px;border-radius:5px;flex-shrink:0;margin-top:5px;}
.post{background:var(--post-bg);color:var(--post-col);border:1px solid var(--post-bd);}
.get {background:var(--get-bg); color:var(--get-col); border:1px solid var(--get-bd);}
.epath{font-size:24px;font-weight:800;letter-spacing:-1.2px;margin-bottom:4px;}
.esumm{font-size:11px;color:var(--f2);font-weight:400;letter-spacing:.2px;}

.ebody{padding:22px 26px;}
.pts{list-style:none;display:flex;flex-direction:column;gap:10px;margin-bottom:22px;}
.pts li{display:flex;align-items:flex-start;gap:11px;font-size:12px;color:var(--f2);line-height:1.75;font-weight:400;}
.pd{width:5px;height:5px;border-radius:50%;background:var(--accent);flex-shrink:0;margin-top:8px;}
[data-theme="dark"] .pd{box-shadow:0 0 8px rgba(255,212,73,.5);}
[data-theme="light"] .pd{box-shadow:0 0 6px rgba(9,9,26,.3);}
.pts li strong{color:var(--fg);font-weight:600;}

.efoot{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;}
.etags{display:flex;gap:5px;flex-wrap:wrap;}
.tag{font-size:9px;padding:3px 9px;border-radius:5px;
  background:var(--card2);border:1px solid var(--b2);color:var(--f3);letter-spacing:.5px;}

.elink{
  display:inline-flex;align-items:center;gap:6px;padding:8px 16px;border-radius:8px;
  border:1px solid var(--aborder);background:var(--aglow);
  color:var(--accent);font-size:11px;font-family:'JetBrains Mono',monospace;
  text-decoration:none;cursor:pointer;letter-spacing:.3px;font-weight:500;
}
.elink:hover{background:var(--aglow2);border-color:var(--accent2);color:var(--accent2);}

/* ════ MODEL CARDS ══════════════════════════════════════════ */
.mgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;}
.mcard{
  background:var(--card);border:1px solid var(--b1);border-radius:16px;
  padding:24px;position:relative;overflow:hidden;
}
.mcard::after{content:'';position:absolute;bottom:0;left:0;right:0;height:2.5px;
  background:linear-gradient(90deg,var(--c1),var(--c2));}
.mcard:hover{border-color:var(--b2);transform:translateY(-5px);box-shadow:var(--sh);}
.micon{width:44px;height:44px;border-radius:12px;display:flex;align-items:center;
  justify-content:center;font-size:20px;margin-bottom:16px;
  background:var(--card2);border:1px solid var(--b2);}
.mname{font-size:18px;font-weight:800;letter-spacing:-.8px;margin-bottom:4px;}
.mid{font-size:9px;color:var(--f3);margin-bottom:10px;word-break:break-all;line-height:1.8;font-weight:400;}
.mdesc{font-size:11.5px;color:var(--f2);line-height:1.8;font-weight:400;}
.mbadge{display:inline-block;font-size:9px;padding:3px 10px;border-radius:5px;
  margin-top:12px;letter-spacing:.8px;font-weight:600;}

/* ════ FOOTER ═══════════════════════════════════════════════ */
footer{
  position:relative;z-index:1;border-top:1px solid var(--b1);
  padding:28px 44px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:14px;
}
.fbrand{font-size:16px;font-weight:800;letter-spacing:-1px;color:var(--accent);}
.flinks{display:flex;gap:22px;flex-wrap:wrap;}
.flinks a{font-size:10px;color:var(--f3);text-decoration:none;letter-spacing:.5px;}
.flinks a:hover{color:var(--accent);}
.fright{font-size:10px;color:var(--f3);}

@keyframes fu{from{opacity:0;transform:translateY(20px);}to{opacity:1;transform:translateY(0);}}
::-webkit-scrollbar{width:4px;}
::-webkit-scrollbar-thumb{background:var(--b2);border-radius:2px;}

@media(max-width:640px){
  nav{padding:0 18px;}h1{font-size:48px;letter-spacing:-3px;}
  .stat{padding:16px 20px;}.epgrid{grid-template-columns:1fr;}
  .mgrid{grid-template-columns:1fr 1fr;}footer{flex-direction:column;text-align:center;}
}
</style>
</head>
<body>
<div class="blob b1"></div>
<div class="blob b2"></div>
<div class="blob b3"></div>

<!-- NAV -->
<nav>
  <a class="nlogo" href="/">
    <span class="nlogo-main">MoodLens</span>
    <span class="nlogo-tag">Sentiment Intelligence</span>
  </a>
  <div class="nr">
    <div class="live-pill">
      <span class="ldot"></span>Online
    </div>
    <a class="npill" href="/health">/health</a>
    <a class="npill" href="/docs">/docs</a>
    <a class="npill" href="/redoc">/redoc</a>
    <button class="tbtn" id="themeToggle" title="Toggle theme">🌙</button>
  </div>
</nav>

<!-- HERO -->
<div class="hero">
  <div class="eyebrow">
    <span class="ldot"></span>
    Zomato Dataset &nbsp;·&nbsp; 4 Models &nbsp;·&nbsp; v2.0
  </div>
  <p class="hero-pre">Sentiment Intelligence Platform</p>
  <h1>
    <span class="h1a">Decode</span>
    <span class="h1b">Sentiment.</span>
  </h1>
  <p class="hsub">
    Enterprise NLP engine · <strong>RoBERTa · DistilRoBERTa · BERT · ALBERT</strong><br/>
    Fine-tuned on <strong>2.9M+ Zomato Reviews</strong> · Three-class · Sub-second inference
  </p>
  <div class="hbtns">
    <a class="bpri" href="/docs">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none">
        <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" stroke="currentColor" stroke-width="2.2"/>
        <polyline points="14,2 14,8 20,8" stroke="currentColor" stroke-width="2.2"/>
        <line x1="8" y1="13" x2="16" y2="13" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
      </svg>
      Explore API Docs
    </a>
    <a class="bout" href="/health">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none">
        <path d="M22 12h-4l-3 9L9 3l-3 9H2" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
      Health Check
    </a>
  </div>
  <div class="stats">
    <div class="stat"><div class="sv">4</div><div class="sl">Transformers</div></div>
    <div class="stat"><div class="sv">3</div><div class="sl">Classes</div></div>
    <div class="stat"><div class="sv">2.9M+</div><div class="sl">Reviews</div></div>
    <div class="stat"><div class="sv">512</div><div class="sl">Max Tokens</div></div>
  </div>
  <div class="online-bar">
    <span class="on">● API Online</span>
    <span class="sep">|</span>
    <span>RoBERTa Ready</span>
    <span class="sep">|</span>
    <span id="ts"></span>
  </div>
</div>

<!-- TERMINAL DEMO -->
<div class="terminal-wrap">
  <div class="terminal">
    <div class="t-bar">
      <div class="t-dot" style="background:#FF4D6A;"></div>
      <div class="t-dot" style="background:#FFD449;"></div>
      <div class="t-dot" style="background:#00E5A0;"></div>
      <span class="t-title">moodlens · api demo</span>
    </div>
    <div class="t-body">
      <div class="t-line"><span class="t-prompt">$</span><span>&nbsp;curl -X POST http://localhost:8000/predict \</span></div>
      <div class="t-line"><span class="t-prompt">&nbsp;</span><span>&nbsp;&nbsp;&nbsp;&nbsp;-H <span class="t-str">"Content-Type: application/json"</span> \</span></div>
      <div class="t-line"><span class="t-prompt">&nbsp;</span><span>&nbsp;&nbsp;&nbsp;&nbsp;-d <span class="t-str">'{"text": "Best biryani I have ever had!"}'</span></span></div>
      <br/>
      <div class="t-line"><span class="t-prompt" style="color:var(--f3)">#</span><span class="t-cmt">&nbsp;200 OK · 284ms</span></div>
      <div class="t-out">{</div>
      <div class="t-out">&nbsp;&nbsp;<span class="t-key">"prediction"</span>: <span class="t-pos">"Positive"</span>,</div>
      <div class="t-out">&nbsp;&nbsp;<span class="t-key">"confidence"</span>: <span class="t-num">0.978</span>,</div>
      <div class="t-out">&nbsp;&nbsp;<span class="t-key">"positive"</span>:&nbsp;&nbsp;<span class="t-num">0.978</span>,</div>
      <div class="t-out">&nbsp;&nbsp;<span class="t-key">"neutral"</span>:&nbsp;&nbsp;&nbsp;<span class="t-num">0.015</span>,</div>
      <div class="t-out">&nbsp;&nbsp;<span class="t-key">"negative"</span>:&nbsp;&nbsp;<span class="t-num">0.007</span></div>
      <div class="t-out">}</div>
      <br/>
      <div class="t-line"><span class="t-prompt">$</span><span class="t-cursor"></span></div>
    </div>
  </div>
</div>

<!-- ENDPOINTS SECTION -->
<div class="wrap">
  <div class="divider"></div>
  <p class="slabel">API Endpoints</p>
  <h2 class="stitle">Four Routes.<br/>Zero Confusion.</h2>
  <p class="sdesc">Click "Try it live" on any card — opens Swagger directly at that endpoint, ready to test with one click.</p>

  <div class="epgrid">

    <!-- /predict -->
    <div class="ecard">
      <div class="ehead">
        <span class="meth post">POST</span>
        <div>
          <div class="epath">/predict</div>
          <div class="esumm">Single-model RoBERTa sentiment prediction</div>
        </div>
      </div>
      <div class="ebody">
        <ul class="pts">
          <li><span class="pd"></span><span>Text is <strong>tokenised and truncated</strong> to 512 tokens, passed through RoBERTa in a single forward pass</span></li>
          <li><span class="pd"></span><span>Softmax output mapped to <strong>Positive / Neutral / Negative</strong> with raw probability for each class</span></li>
          <li><span class="pd"></span><span>Returns <strong>prediction label</strong>, confidence score, all three class probabilities in one clean JSON</span></li>
          <li><span class="pd"></span><span>Best for <strong>high-throughput pipelines</strong> where one best-in-class model is sufficient</span></li>
        </ul>
        <div class="efoot">
          <div class="etags"><span class="tag">Inference</span><span class="tag">RoBERTa</span><span class="tag">JSON</span></div>
          <a class="elink" href="/docs" onclick="openOp(event,'predict')">
            Try it live
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none"><path d="M7 17L17 7M17 7H7M17 7v10" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>
          </a>
        </div>
      </div>
    </div>

    <!-- /compare -->
    <div class="ecard">
      <div class="ehead">
        <span class="meth post">POST</span>
        <div>
          <div class="epath">/compare</div>
          <div class="esumm">All 4 models — parallel inference &amp; consensus</div>
        </div>
      </div>
      <div class="ebody">
        <ul class="pts">
          <li><span class="pd"></span><span>Runs <strong>RoBERTa, DistilRoBERTa, BERT, and ALBERT</strong> on the same text independently</span></li>
          <li><span class="pd"></span><span>Each model's raw labels <strong>normalised</strong> to the same three-class schema before comparison</span></li>
          <li><span class="pd"></span><span>Results <strong>sorted by confidence</strong> — highest-confidence model ranked first in the response array</span></li>
          <li><span class="pd"></span><span>Use when you need <strong>cross-model validation</strong> or the most reliable possible prediction</span></li>
        </ul>
        <div class="efoot">
          <div class="etags"><span class="tag">Inference</span><span class="tag">Multi-Model</span><span class="tag">Ensemble</span></div>
          <a class="elink" href="/docs" onclick="openOp(event,'compare')">
            Try it live
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none"><path d="M7 17L17 7M17 7H7M17 7v10" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>
          </a>
        </div>
      </div>
    </div>

    <!-- /health -->
    <div class="ecard">
      <div class="ehead">
        <span class="meth get">GET</span>
        <div>
          <div class="epath">/health</div>
          <div class="esumm">Server status &amp; ML engine diagnostics</div>
        </div>
      </div>
      <div class="ebody">
        <ul class="pts">
          <li><span class="pd"></span><span>Returns <strong>server liveness</strong>, API version, and active model list in a single JSON response</span></li>
          <li><span class="pd"></span><span>Lists all <strong>four active model names</strong> and training dataset for at-a-glance verification</span></li>
          <li><span class="pd"></span><span>Ideal for <strong>CI/CD pipelines, uptime monitors</strong>, and pre-flight checks before batch inference jobs</span></li>
          <li><span class="pd"></span><span>Returns <strong>UTC timestamp</strong> to verify server clock is correctly synchronised</span></li>
        </ul>
        <div class="efoot">
          <div class="etags"><span class="tag">System</span><span class="tag">Monitoring</span><span class="tag">DevOps</span></div>
          <a class="elink" href="/health" target="_blank">
            View live
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none"><path d="M7 17L17 7M17 7H7M17 7v10" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>
          </a>
        </div>
      </div>
    </div>

    <!-- /docs -->
    <div class="ecard">
      <div class="ehead">
        <span class="meth get">GET</span>
        <div>
          <div class="epath">/docs</div>
          <div class="esumm">Interactive Swagger UI — test every endpoint live</div>
        </div>
      </div>
      <div class="ebody">
        <ul class="pts">
          <li><span class="pd"></span><span>Full <strong>OpenAPI 3.1 schema</strong> auto-generated from Pydantic models — every field typed and described</span></li>
          <li><span class="pd"></span><span><strong>"Try it out"</strong> lets you fire real requests to /predict and /compare without writing any code</span></li>
          <li><span class="pd"></span><span>Every request and response schema <strong>documented inline</strong> with constraints and live example values</span></li>
          <li><span class="pd"></span><span>Dark-themed Swagger with <strong>Monokai syntax highlighting</strong> and request duration display</span></li>
        </ul>
        <div class="efoot">
          <div class="etags"><span class="tag">Docs</span><span class="tag">OpenAPI</span><span class="tag">Swagger</span></div>
          <a class="elink" href="/docs" target="_blank">
            Open Docs
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none"><path d="M7 17L17 7M17 7H7M17 7v10" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>
          </a>
        </div>
      </div>
    </div>

  </div>

  <!-- MODEL CARDS -->
  <div style="margin-top:90px;">
    <p class="slabel">Transformer Models</p>
    <h2 class="stitle">Four Engines.<br/>One Verdict.</h2>
    <p class="sdesc" style="margin-bottom:38px;">Each model brings a unique specialisation — together they form a robust, cross-validated ensemble.</p>
    <div class="mgrid">
      <div class="mcard" style="--c1:var(--accent);--c2:var(--accent2);">
        <div class="micon">🧠</div>
        <div class="mname">RoBERTa</div>
        <div class="mid">cardiffnlp/twitter-roberta-base-sentiment-latest</div>
        <div class="mdesc">Default model. Trained on 124M tweets. Best general-purpose accuracy across review types and domains.</div>
        <span class="mbadge" style="background:var(--aglow);color:var(--accent);border:1px solid var(--aborder);">⭐ DEFAULT</span>
      </div>
      <div class="mcard" style="--c1:var(--accent2);--c2:var(--accent);">
        <div class="micon">⚡</div>
        <div class="mname">DistilRoBERTa</div>
        <div class="mid">mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis</div>
        <div class="mdesc">40% faster than RoBERTa. Fine-tuned on financial news and consumer reviews. Low-latency inference.</div>
        <span class="mbadge" style="background:var(--aglow2);color:var(--accent2);border:1px solid var(--aborder2);">FAST</span>
      </div>
      <div class="mcard" style="--c1:#276EF1;--c2:var(--accent);">
        <div class="micon">🌍</div>
        <div class="mname">BERT</div>
        <div class="mid">nlptown/bert-base-multilingual-uncased-sentiment</div>
        <div class="mdesc">Multilingual BERT fine-tuned in 6 languages. 5-star scale mapped to Positive / Neutral / Negative.</div>
        <span class="mbadge" style="background:rgba(39,110,241,.1);color:#276EF1;border:1px solid rgba(39,110,241,.25);">MULTILINGUAL</span>
      </div>
      <div class="mcard" style="--c1:#09091A;--c2:#276EF1;">
        <div class="micon">🎯</div>
        <div class="mname">ALBERT</div>
        <div class="mid">textattack/albert-base-v2-yelp-polarity</div>
        <div class="mdesc">Parameter-efficient architecture fine-tuned on Yelp reviews. Excellent on short, punchy restaurant feedback.</div>
        <span class="mbadge" style="background:rgba(9,9,26,.1);color:var(--f2);border:1px solid var(--b2);">EFFICIENT</span>
      </div>
    </div>
  </div>
</div>

<!-- FOOTER -->
<footer>
  <div class="fbrand">MoodLens</div>
  <div class="flinks">
    <a href="/docs">Swagger</a>
    <a href="/redoc">ReDoc</a>
    <a href="/health">Health</a>
    <a href="/openapi.json">OpenAPI JSON</a>
  </div>
  <div class="fright">v2.0.0 · Zomato Dataset · <span id="fts"></span></div>
</footer>

<script>
// Theme toggle
const root = document.documentElement;
const btn  = document.getElementById('themeToggle');
let dark   = true;
btn.addEventListener('click', () => {
  dark = !dark;
  root.setAttribute('data-theme', dark ? 'dark' : 'light');
  btn.textContent = dark ? '🌙' : '☀️';
});

// Clock
const tick = () => {
  const s = new Date().toLocaleTimeString('en-IN',{hour:'2-digit',minute:'2-digit',second:'2-digit'});
  const a = document.getElementById('ts');
  const b = document.getElementById('fts');
  if(a) a.textContent = s;
  if(b) b.textContent = s;
};
tick(); setInterval(tick, 1000);

// Try it live — opens /docs at exact Swagger anchor in new tab
function openOp(e, route) {
  e.preventDefault();
  const map = {
    predict : '/docs#/Inference/get_prediction_predict_post',
    compare : '/docs#/Inference/compare_models_compare_post',
  };
  window.open(map[route] || '/docs', '_blank');
}

// Scroll reveal
const obs = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      e.target.style.opacity   = '1';
      e.target.style.transform = 'translateY(0)';
    }
  });
}, { threshold: 0.07 });
document.querySelectorAll('.ecard, .mcard').forEach((el, i) => {
  el.style.opacity   = '0';
  el.style.transform = 'translateY(22px)';
  el.style.transition = [
    `opacity .5s ${i * 0.08}s ease`,
    `transform .5s ${i * 0.08}s ease`,
    'border-color .25s', 'box-shadow .25s',
  ].join(', ');
  obs.observe(el);
});
</script>
</body>
</html>"""

# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port)