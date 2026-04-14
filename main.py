from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import snntorch as snn
import snntorch.functional as SF
from datetime import datetime
import json
from pathlib import Path
import trafilatura
import textstat
import anthropic
import os
import io
import base64
import tempfile
import statistics
import re
from typing import Optional, List
import fitz  # PyMuPDF

app = FastAPI(title="AIRIA SNN Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

claude_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


# ─────────────────────────────────────────────
# SNN MODEL
# ─────────────────────────────────────────────

class AiriaSNN(nn.Module):
    def __init__(self):
        super().__init__()
        beta = 0.95
        self.fc1 = nn.Linear(6, 16)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(16, 8)
        self.lif2 = snn.Leaky(beta=beta)
        self.fc3 = nn.Linear(8, 3)
        self.lif3 = snn.Leaky(beta=beta)

    def forward_snapshot(self, x, num_steps=25):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        spk3_rec = []
        for _ in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3_rec.append(spk3)
        return torch.stack(spk3_rec)

    def forward_step(self, x, mem1, mem2, mem3):
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)
        cur2 = self.fc2(spk1)
        spk2, mem2 = self.lif2(cur2, mem2)
        cur3 = self.fc3(spk2)
        spk3, mem3 = self.lif3(cur3, mem3)
        return spk3, mem1, mem2, mem3


# ─────────────────────────────────────────────
# BASE TRAINING DATA
#
# Feature order: [avg_wpm, wpm_variance, back_presses, completion_rate, slowdown_ratio, blur_count]
#
# Normalization:
#   avg_wpm        = raw_wpm / 500
#   wpm_variance   = std_dev / 200
#   back_presses   = count / 10
#   completion_rate = fraction 0-1
#   slowdown_ratio = avg_wpm / this_para_wpm, clamped 0-1
#                    (1.0 = this paragraph was much slower than average = struggle)
#   blur_count     = tab_aways / 5
#
# Label mapping (must match spike_classes in predict_paragraph):
#   0 = too_hard
#   1 = just_right
#   2 = too_easy
# ─────────────────────────────────────────────

X_train = torch.tensor([
    # ── too_hard (label 0): slow absolute WPM, high slowdown, high variance ──
    [0.36,    0.55,  0.60,  0.45,  1.00,  0.40],  # 180 WPM, big slowdown
    [0.40,    0.60,  0.50,  0.50,  1.00,  0.50],  # 200 WPM, high variance
    [0.32,    0.65,  0.70,  0.40,  1.00,  0.60],  # 160 WPM, lots of back presses
    [0.38,    0.50,  0.40,  0.55,  0.95,  0.30],  # 190 WPM, moderate struggle
    [0.34,    0.70,  0.80,  0.35,  1.00,  0.70],  # 170 WPM, high distraction
    [0.42,    0.45,  0.30,  0.60,  0.90,  0.20],  # 210 WPM, mild struggle

    # ── just_right (label 1): mid WPM, low variance, minimal friction ────────
    [0.58,    0.15,  0.10,  0.85,  0.65,  0.05],  # 290 WPM, comfortable
    [0.54,    0.20,  0.20,  0.80,  0.70,  0.10],  # 270 WPM, slight variance
    [0.62,    0.18,  0.10,  0.90,  0.60,  0.05],  # 310 WPM, smooth
    [0.56,    0.22,  0.10,  0.85,  0.68,  0.08],  # 280 WPM, normal
    [0.60,    0.16,  0.20,  0.88,  0.62,  0.06],  # 300 WPM, low distraction
    [0.52,    0.25,  0.30,  0.75,  0.75,  0.12],  # 260 WPM, few back presses

    # ── too_easy (label 2): fast WPM, near-zero variance, no friction ────────
    [0.80,    0.05,  0.00,  1.00,  0.30,  0.00],  # 400 WPM, skimming
    [0.86,    0.04,  0.00,  1.00,  0.25,  0.00],  # 430 WPM, very fast
    [0.76,    0.06,  0.00,  1.00,  0.35,  0.00],  # 380 WPM, breezing
    [0.90,    0.03,  0.00,  1.00,  0.20,  0.00],  # 450 WPM, near max
    [0.78,    0.07,  0.00,  0.98,  0.32,  0.00],  # 390 WPM
    [0.82,    0.05,  0.00,  1.00,  0.28,  0.00],  # 410 WPM, zero friction
], dtype=torch.float32)

# Labels explicitly named for clarity — 0=too_hard, 1=just_right, 2=too_easy
y_train = torch.tensor([
    0, 0, 0, 0, 0, 0,   # too_hard
    1, 1, 1, 1, 1, 1,   # just_right
    2, 2, 2, 2, 2, 2,   # too_easy
])

# spike_classes list in predict_paragraph must match this order exactly
SPIKE_CLASSES = ["too_hard", "just_right", "too_easy"]


# ─────────────────────────────────────────────
# WEIGHT + MEMBRANE HELPERS
# ─────────────────────────────────────────────

def weights_to_base64(model: AiriaSNN) -> str:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def base64_to_weights(b64: str) -> dict:
    buf = io.BytesIO(base64.b64decode(b64))
    return torch.load(buf, weights_only=True)


def membrane_to_list(mem: torch.Tensor) -> list:
    # Flatten [1, N] → [N] before serializing so Pydantic gets List[float]
    return mem.detach().flatten().tolist()


def list_to_membrane(data: list) -> torch.Tensor:
    # Restore [1, N] batch dim that snntorch expects
    return torch.tensor(data, dtype=torch.float32).unsqueeze(0)


def fresh_model() -> AiriaSNN:
    model = AiriaSNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
    for epoch in range(200):
        model.train()
        spk_out = model.forward_snapshot(X_train)
        loss = loss_fn(spk_out, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def load_model_from_b64(b64: Optional[str]) -> AiriaSNN:
    model = AiriaSNN()
    if b64:
        try:
            model.load_state_dict(base64_to_weights(b64))
        except Exception as e:
            print(f"Failed to load user weights, falling back to base model: {e}")
            model = fresh_model()
    else:
        model = fresh_model()
    return model


print("Warming base model...")
_base_model = fresh_model()
BASE_WEIGHTS_B64 = weights_to_base64(_base_model)
print("Base model ready.")


# ─────────────────────────────────────────────
# PYDANTIC MODELS
# ─────────────────────────────────────────────

class TrainingSample(BaseModel):
    features: List[float]
    label: int
    timestamp: str = ""

class ParagraphFeatures(BaseModel):
    avg_wpm: float
    wpm_variance: float
    back_presses: float
    completion_rate: float
    slowdown_ratio: float
    blur_count: float

class PredictParagraphRequest(BaseModel):
    features: ParagraphFeatures
    weights: Optional[str] = None
    mem1: Optional[List[float]] = None
    mem2: Optional[List[float]] = None
    mem3: Optional[List[float]] = None

class PredictParagraphResponse(BaseModel):
    spiked: bool
    spike_class: str
    spike_values: List[float]
    mem1: List[float]
    mem2: List[float]
    mem3: List[float]
    membrane_charge: float

class PredictRequest(BaseModel):
    avg_wpm: float
    wpm_variance: float
    back_presses: float
    completion_rate: float
    slowdown_ratio: float
    blur_count: float
    weights: Optional[str] = None

class PredictionResponse(BaseModel):
    action: str
    confidence: float
    raw_scores: list

class RetrainRequest(BaseModel):
    samples: List[TrainingSample]
    weights: Optional[str] = None

class RetrainResponse(BaseModel):
    status: str
    weights: str
    user_samples: int
    total_samples: int
    current_samples: int = 0
    required: int = 0

class BaseWeightsResponse(BaseModel):
    weights: str

class IngestURLRequest(BaseModel):
    url: str

class IngestURLResponse(BaseModel):
    status: str
    title: str
    content: str
    estimated_lexile: int
    word_count: int
    paragraph_count: int
    article_id: str
    classification: dict = {}

class InterventionRequest(BaseModel):
    paragraph: str
    genre_difficulty: float
    specific_genre: str = "general"

class InterventionResponse(BaseModel):
    level: int
    rewrite_strength: str
    primer: str
    rewritten: str
    annotation: str

class AnnotationTerm(BaseModel):
    term: str
    definition: str
    start: int
    end: int

class AnnotateRequest(BaseModel):
    paragraph: str
    specific_genre: str = "general"
    genre_difficulty: float = 0.5

class AnnotateResponse(BaseModel):
    terms: List[AnnotationTerm]

class CalibrationArticle(BaseModel):
    # "too_hard", "just_right", "comfortable", or "too_easy"
    difficulty_rating: str
    avg_wpm: float

class CalibrateRequest(BaseModel):
    articles: List[CalibrationArticle]

class CalibrateResponse(BaseModel):
    charge_threshold: float
    slowdown_threshold: float

class ExtractPDFResponse(BaseModel):
    text: str
    page_count: int


# ─────────────────────────────────────────────
# CLASSIFICATION
# ─────────────────────────────────────────────

def classify_text_with_claude(text: str) -> dict:
    words = text.split()
    sample = " ".join(words[:1500])

    prompt = f"""You are a text classification system. Analyze the following article excerpt and return a JSON object with exactly these four fields:

1. "broad_genre": one of ["science", "technology", "politics", "geopolitics", "culture", "history", "economics", "health", "environment", "philosophy", "sports", "other"]
2. "specific_genre": a more specific label within that broad genre, e.g. "aerospace engineering", "monetary policy", "ancient Rome", "machine learning" — be precise
3. "genre_difficulty": a float from 0.0 to 1.0 representing how cognitively dense this genre typically is for a general reader. Use this scale:
   - 0.0 to 0.2: casual narrative, lifestyle, sports recaps
   - 0.3 to 0.5: general news, politics, culture commentary
   - 0.6 to 0.8: technical journalism, economics analysis, hard science reporting
   - 0.9 to 1.0: academic papers, dense policy analysis, highly specialized technical content
4. "reasoning": one sentence explaining your classification

Respond with ONLY valid JSON, no markdown, no explanation outside the JSON.

Article excerpt:
{sample}"""

    message = claude_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = message.content[0].text.strip()
    classification = json.loads(raw)
    classification["genre_difficulty"] = max(0.0, min(1.0, float(classification["genre_difficulty"])))
    return classification


# ─────────────────────────────────────────────
# SCRAPING HELPERS
# ─────────────────────────────────────────────

UNIVERSAL_SKIP = [
    'subscribe', 'sign up for', 'newsletter', 'share this',
    'follow us on', 'image:', 'photo credit:', 'advertisement',
    'all rights reserved', 'terms of service', 'privacy policy',
]

def clean_paragraphs(raw_text: str) -> list:
    paragraphs = raw_text.split('\n\n')
    cleaned = []
    for para in paragraphs:
        para = ' '.join(para.split()).strip()
        if len(para.split()) < 8:
            continue
        if any(p in para.lower() for p in UNIVERSAL_SKIP):
            continue
        cleaned.append(para)
    return cleaned


def scrape_with_newspaper(url: str) -> tuple:
    from newspaper import Article
    article = Article(url)
    article.download()
    article.parse()
    if not article.text:
        raise ValueError("newspaper3k returned empty content")
    title = article.title or "Untitled Article"
    return title, article.text


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "status": "AIRIA SNN Backend Running",
        "version": "3.7",
        "storage": "stateless — all user data in browser localStorage",
        "snn_mode": "temporal per-paragraph + end-of-session snapshot"
    }


@app.get("/base-weights", response_model=BaseWeightsResponse)
async def get_base_weights():
    return BaseWeightsResponse(weights=BASE_WEIGHTS_B64)


@app.post("/predict-paragraph", response_model=PredictParagraphResponse)
async def predict_paragraph(data: PredictParagraphRequest):
    model = load_model_from_b64(data.weights)
    model.eval()

    x = torch.tensor([[
        data.features.avg_wpm,
        data.features.wpm_variance,
        data.features.back_presses,
        data.features.completion_rate,
        data.features.slowdown_ratio,
        data.features.blur_count
    ]], dtype=torch.float32)

    mem1 = list_to_membrane(data.mem1) if data.mem1 is not None else model.lif1.init_leaky()
    mem2 = list_to_membrane(data.mem2) if data.mem2 is not None else model.lif2.init_leaky()
    mem3 = list_to_membrane(data.mem3) if data.mem3 is not None else model.lif3.init_leaky()

    with torch.no_grad():
        spk3, mem1, mem2, mem3 = model.forward_step(x, mem1, mem2, mem3)

    spike_values = spk3.squeeze().tolist()
    if isinstance(spike_values, float):
        spike_values = [spike_values, 0.0, 0.0]

    # Index 0 = too_hard, 1 = just_right, 2 = too_easy — must match y_train labels
    spiked = any(v > 0.5 for v in spike_values)
    spike_class_idx = int(torch.tensor(spike_values).argmax().item())

    mem3_flat = mem3.squeeze()
    if mem3_flat.dim() == 0:
        membrane_charge = float(torch.sigmoid(mem3_flat).item())
    else:
        membrane_charge = float(torch.sigmoid(mem3_flat).mean().item())

    return PredictParagraphResponse(
        spiked=spiked,
        spike_class=SPIKE_CLASSES[spike_class_idx],
        spike_values=spike_values,
        mem1=membrane_to_list(mem1),
        mem2=membrane_to_list(mem2),
        mem3=membrane_to_list(mem3),
        membrane_charge=membrane_charge
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(data: PredictRequest):
    model = load_model_from_b64(data.weights)

    x = torch.tensor([[
        data.avg_wpm, data.wpm_variance, data.back_presses,
        data.completion_rate, data.slowdown_ratio, data.blur_count
    ]], dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        spk_out = model.forward_snapshot(x)
        spike_counts = spk_out.sum(dim=0).squeeze()
        prediction = spike_counts.argmax().item()

    return PredictionResponse(
        action=SPIKE_CLASSES[prediction],
        confidence=float(spike_counts.max()),
        raw_scores=spike_counts.tolist()
    )


@app.post("/retrain", response_model=RetrainResponse)
async def retrain(data: RetrainRequest):
    MIN_SAMPLES = 10

    if len(data.samples) < MIN_SAMPLES:
        return RetrainResponse(
            status="insufficient_data",
            weights=BASE_WEIGHTS_B64,
            user_samples=len(data.samples),
            total_samples=0,
            current_samples=len(data.samples),
            required=MIN_SAMPLES
        )

    model = load_model_from_b64(data.weights)

    user_X = torch.tensor([s.features for s in data.samples], dtype=torch.float32)
    user_y = torch.tensor([s.label for s in data.samples])
    combined_X = torch.cat([X_train, user_X, user_X])
    combined_y = torch.cat([y_train, user_y, user_y])

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

    for epoch in range(100):
        model.train()
        spk_out = model.forward_snapshot(combined_X)
        loss = loss_fn(spk_out, combined_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 25 == 0:
            print(f"Retrain Epoch {epoch+1}, Loss: {loss.item():.4f}")

    return RetrainResponse(
        status="retrained",
        weights=weights_to_base64(model),
        user_samples=len(data.samples),
        total_samples=len(combined_X)
    )


@app.post("/intervene", response_model=InterventionResponse)
async def intervene(data: InterventionRequest):
    level = 3 if data.genre_difficulty > 0.7 else 2
    rewrite_strength = "aggressive" if level == 3 else "moderate"

    primer_prompt = f"""You are a reading assistant. A user is struggling with a paragraph from a {data.specific_genre} article.

Write a short background knowledge card — 2 to 4 sentences — that gives the reader just enough context to understand the paragraph. Do not summarize the paragraph. Explain the underlying concept or domain knowledge it assumes the reader already has.

Original paragraph:
{data.paragraph}

Respond with a JSON object with exactly two fields:
1. "primer": the background knowledge card as a plain string (2-4 sentences, no jargon, plain language)
2. "annotation": one sentence identifying the primary knowledge gap

Respond with ONLY valid JSON, no markdown."""

    if level == 2:
        rewrite_prompt = f"""Rewrite the following paragraph from a {data.specific_genre} article at a moderate simplification level. Preserve the original structure. Use simpler vocabulary and break up long sentences. Do not add new information or remove key facts.

Original paragraph:
{data.paragraph}

Respond with a JSON object with exactly one field:
1. "rewritten": the simplified paragraph as a plain string

Respond with ONLY valid JSON, no markdown."""
    else:
        rewrite_prompt = f"""Rewrite the following paragraph from a {data.specific_genre} article at an aggressive simplification level. Reconstruct it entirely for maximum clarity at roughly an 8th grade reading level. Do not preserve original structure. Keep all key facts.

Original paragraph:
{data.paragraph}

Respond with a JSON object with exactly one field:
1. "rewritten": the reconstructed paragraph as a plain string

Respond with ONLY valid JSON, no markdown."""

    try:
        primer_msg    = claude_client.messages.create(model="claude-sonnet-4-20250514", max_tokens=512, messages=[{"role": "user", "content": primer_prompt}])
        primer_result = json.loads(primer_msg.content[0].text.strip())

        rewrite_msg    = claude_client.messages.create(model="claude-sonnet-4-20250514", max_tokens=1024, messages=[{"role": "user", "content": rewrite_prompt}])
        rewrite_result = json.loads(rewrite_msg.content[0].text.strip())

        return InterventionResponse(
            level=level,
            rewrite_strength=rewrite_strength,
            primer=primer_result["primer"],
            rewritten=rewrite_result["rewritten"],
            annotation=primer_result.get("annotation", "")
        )
    except Exception as e:
        print(f"Intervention failed: {e}")
        return InterventionResponse(
            level=level,
            rewrite_strength=rewrite_strength,
            primer="Background context unavailable for this paragraph.",
            rewritten=data.paragraph,
            annotation="Intervention unavailable — showing original paragraph."
        )


@app.post("/annotate", response_model=AnnotateResponse)
async def annotate(data: AnnotateRequest):
    difficulty_instruction = (
        "Include conceptual explanations, not just terms. Prioritize clarity over brevity."
        if data.genre_difficulty > 0.7 else
        "Only annotate uncommon or specialized terms — skip anything a general reader would know."
    )

    prompt = f"""Analyze this paragraph from a {data.specific_genre} article (difficulty: {data.genre_difficulty:.2f}/1.0).

Your goal is to identify what a general reader would struggle to understand in this specific paragraph. Focus on three types of annotation targets:
1. Domain-specific terms (technical vocabulary)
2. Article-specific references (events, policies, organizations, people mentioned without explanation)
3. Implicit concepts the paragraph assumes the reader already knows

For each item provide a plain-language definition — one sentence, under 20 words. Only include items that genuinely block comprehension. Avoid obvious words and general academic vocabulary. Return 0 to 6 total annotations, prioritizing the most important blockers.

{difficulty_instruction}

For each annotation, include the exact character position in the original paragraph text (zero-based start and end index).

Paragraph:
{data.paragraph}

Respond with ONLY valid JSON:
{{"terms": [{{"term": "...", "definition": "...", "start": 0, "end": 0}}]}}"""

    try:
        message = claude_client.messages.create(model="claude-sonnet-4-20250514", max_tokens=1024, messages=[{"role": "user", "content": prompt}])
        result  = json.loads(message.content[0].text.strip())

        para_len = len(data.paragraph)
        terms = []
        for t in result.get("terms", []):
            start = max(0, min(int(t["start"]), para_len))
            end   = max(start, min(int(t["end"]), para_len))
            if data.paragraph[start:end].lower() != t["term"].lower():
                idx = data.paragraph.lower().find(t["term"].lower())
                if idx == -1:
                    continue
                start, end = idx, idx + len(t["term"])
            terms.append(AnnotationTerm(term=t["term"], definition=t["definition"], start=start, end=end))

        return AnnotateResponse(terms=terms)
    except Exception as e:
        print(f"Annotation failed: {e}")
        return AnnotateResponse(terms=[])


@app.post("/ingest-url", response_model=IngestURLResponse)
async def ingest_url(data: IngestURLRequest):
    try:
        downloaded = trafilatura.fetch_url(data.url)
        content = None
        title   = "Untitled Article"

        if downloaded:
            content  = trafilatura.extract(downloaded, include_comments=False, include_tables=False, include_images=False, no_fallback=False, favor_precision=False, favor_recall=True, include_formatting=True)
            metadata = trafilatura.extract_metadata(downloaded)
            if metadata and metadata.title:
                title = metadata.title

        if not content:
            print(f"trafilatura empty for {data.url}, trying newspaper3k...")
            try:
                title, content = scrape_with_newspaper(data.url)
            except Exception as ne:
                raise ValueError(f"Both scrapers failed — trafilatura: empty, newspaper3k: {ne}")

        cleaned_paragraphs = clean_paragraphs(content)
        if not cleaned_paragraphs:
            raise ValueError("Content extracted but all paragraphs were filtered out")

        clean_content    = '\n\n'.join(cleaned_paragraphs)
        word_count       = len(clean_content.split())
        paragraph_count  = len(cleaned_paragraphs)
        fk_grade         = textstat.flesch_kincaid_grade(clean_content)
        estimated_lexile = max(200, min(1600, int(fk_grade * 100 + 200)))

        classification = {}
        try:
            classification = classify_text_with_claude(clean_content)
        except Exception as ce:
            print(f"Classification failed: {ce}")
            classification = {"broad_genre": "other", "specific_genre": "unknown", "genre_difficulty": 0.5, "reasoning": "Classification unavailable"}

        article_id = f"article_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

        return IngestURLResponse(status="success", title=title, content=clean_content, estimated_lexile=estimated_lexile, word_count=word_count, paragraph_count=paragraph_count, article_id=article_id, classification=classification)

    except Exception as e:
        return IngestURLResponse(status="error", title="Error", content=str(e), estimated_lexile=0, word_count=0, paragraph_count=0, article_id="", classification={})


@app.post("/extract-pdf", response_model=ExtractPDFResponse)
async def extract_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        doc = fitz.open(tmp_path)
        page_count = doc.page_count

        raw_blocks = []
        for page in doc:
            for block in page.get_text("blocks"):
                # block: (x0, y0, x1, y1, text, block_no, block_type); type 0 = text
                if block[6] == 0:
                    raw_blocks.append(block[4])
        doc.close()

        raw_text = "\n".join(raw_blocks)

        # Rejoin hyphenated line breaks (e.g. "compre-\nhension" → "comprehension")
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', raw_text)
        # Strip lines that are lone page numbers
        text = re.sub(r'(?m)^\s*\d{1,4}\s*$', '', text)
        # Collapse single newlines within paragraphs into spaces; preserve double newlines
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        # Collapse runs of spaces
        text = re.sub(r' {2,}', ' ', text)
        text = text.strip()

        if len(text) < 100:
            raise HTTPException(status_code=422, detail="PDF appears to be scanned. OCR not supported.")

        return ExtractPDFResponse(text=text, page_count=page_count)

    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


@app.post("/calibrate", response_model=CalibrateResponse)
async def calibrate(data: CalibrateRequest):
    """
    Called after the 3rd calibration article. Derives per-user SNN thresholds
    from article difficulty ratings and WPM spread, then returns them for the
    frontend to persist in airia_user_profile (localStorage).
    """
    COMFORTABLE_RATINGS = {"comfortable", "just_right", "too_easy"}

    too_hard_count  = sum(1 for a in data.articles if a.difficulty_rating == "too_hard")
    comfortable_count = sum(1 for a in data.articles if a.difficulty_rating in COMFORTABLE_RATINGS)

    if too_hard_count >= 2:
        charge_threshold = 0.55
    elif comfortable_count >= 2:
        charge_threshold = 0.78
    else:
        charge_threshold = 0.68

    wpms = [a.avg_wpm for a in data.articles]
    wpm_std = statistics.stdev(wpms) if len(wpms) >= 2 else 0.0

    if wpm_std > 60:
        slowdown_threshold = 0.82
    elif wpm_std < 25:
        slowdown_threshold = 0.68
    else:
        slowdown_threshold = 0.75

    return CalibrateResponse(
        charge_threshold=charge_threshold,
        slowdown_threshold=slowdown_threshold
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)