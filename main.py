from fastapi import FastAPI
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
from typing import Optional, List

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
        """
        Snapshot mode: runs num_steps internal timesteps on a single
        feature vector. Used for end-of-session classification and for
        training, where we need stable gradients over multiple steps.
        The 25 steps here are simulation steps, not real time.
        """
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
        """
        Temporal mode: processes ONE paragraph as ONE real timestep.
        Membrane state is passed in from the frontend and returned updated.

        This is the correct SNN usage. Each paragraph is a real event in
        time. The membrane accumulates genuine reading struggle across the
        article. Early struggle primes the membrane so later struggle
        triggers faster. The frontend holds mem1/mem2/mem3 in JavaScript
        memory between paragraphs and discards them when the session ends.

        No artificial internal timesteps. One paragraph = one step.
        """
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)
        cur2 = self.fc2(spk1)
        spk2, mem2 = self.lif2(cur2, mem2)
        cur3 = self.fc3(spk2)
        spk3, mem3 = self.lif3(cur3, mem3)
        return spk3, mem1, mem2, mem3


# Base training data
X_train = torch.tensor([
    [0.2, 0.8, 0.7, 0.6, 0.3, 0.6],
    [0.25, 0.7, 0.6, 0.7, 0.4, 0.5],
    [0.15, 0.9, 0.8, 0.5, 0.2, 0.7],
    [0.3, 0.6, 0.5, 0.8, 0.35, 0.4],
    [0.18, 0.85, 0.7, 0.55, 0.25, 0.65],
    [0.22, 0.75, 0.65, 0.65, 0.3, 0.55],
    [0.5, 0.3, 0.2, 0.95, 0.7, 0.1],
    [0.55, 0.25, 0.15, 1.0, 0.75, 0.05],
    [0.45, 0.35, 0.25, 0.9, 0.65, 0.15],
    [0.5, 0.3, 0.1, 1.0, 0.7, 0.1],
    [0.48, 0.32, 0.2, 0.95, 0.68, 0.12],
    [0.52, 0.28, 0.18, 1.0, 0.72, 0.08],
    [0.8, 0.15, 0.05, 1.0, 0.85, 0.0],
    [0.85, 0.1, 0.0, 1.0, 0.9, 0.0],
    [0.75, 0.2, 0.1, 1.0, 0.8, 0.05],
    [0.9, 0.1, 0.0, 1.0, 0.92, 0.0],
    [0.78, 0.18, 0.08, 1.0, 0.82, 0.02],
    [0.82, 0.12, 0.05, 1.0, 0.88, 0.0],
], dtype=torch.float32)
y_train = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])


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
    """Serialize membrane tensor to plain list for JSON transport."""
    return mem.detach().tolist()


def list_to_membrane(data: list) -> torch.Tensor:
    """Deserialize membrane state from JSON back to tensor."""
    return torch.tensor(data, dtype=torch.float32)


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
    """
    Per-paragraph temporal inference.
    mem1/mem2/mem3 carry the membrane state from the previous paragraph.
    Null on the first paragraph of a session — fresh zeros are used.
    """
    features: ParagraphFeatures
    weights: Optional[str] = None
    mem1: Optional[List[float]] = None
    mem2: Optional[List[float]] = None
    mem3: Optional[List[float]] = None

class PredictParagraphResponse(BaseModel):
    """
    Returns spike output plus updated membrane states.
    Frontend carries mem1/mem2/mem3 into the next paragraph call.
    membrane_charge is a 0-1 scalar for a live difficulty gauge.
    """
    spiked: bool
    spike_class: str            # too_hard / just_right / too_easy
    spike_values: List[float]
    mem1: List[float]
    mem2: List[float]
    mem3: List[float]
    membrane_charge: float      # how charged the output layer is overall

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

# ── NEW: Intervention models ──────────────────────────────────────────────────

class InterventionRequest(BaseModel):
    paragraph: str              # the paragraph the user struggled with
    genre_difficulty: float     # 0.0-1.0 from article classification
    specific_genre: str = "general"

class InterventionResponse(BaseModel):
    level: int                  # 2 = full rewrite, 3 = surgical annotation
    rewritten: str              # the altered paragraph shown to the user
    annotation: str             # one-sentence explanation of what was hard


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
# ROUTES
# ─────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "status": "AIRIA SNN Backend Running",
        "version": "3.2",
        "storage": "stateless — all user data in browser localStorage",
        "snn_mode": "temporal per-paragraph + end-of-session snapshot"
    }


@app.get("/base-weights", response_model=BaseWeightsResponse)
async def get_base_weights():
    return BaseWeightsResponse(weights=BASE_WEIGHTS_B64)


@app.post("/predict-paragraph", response_model=PredictParagraphResponse)
async def predict_paragraph(data: PredictParagraphRequest):
    """
    Temporal inference — called after each paragraph is read.

    The frontend passes this paragraph's 6 features plus the membrane
    states from the previous paragraph. The SNN runs exactly one real
    timestep. Updated membrane states are returned for the next call.

    This is the correct SNN usage pattern:
    - Each paragraph = one real timestep in the network
    - Membrane accumulates genuine struggle across the article
    - Early struggle primes the system, later struggle fires faster
    - No artificial internal simulation steps
    - Server holds nothing between requests
    """
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

    spiked = any(v > 0.5 for v in spike_values)
    spike_class_idx = int(torch.tensor(spike_values).argmax().item())
    spike_classes = ["too_hard", "just_right", "too_easy"]

    mem3_flat = mem3.squeeze()
    if mem3_flat.dim() == 0:
        membrane_charge = float(torch.sigmoid(mem3_flat).item())
    else:
        membrane_charge = float(torch.sigmoid(mem3_flat).mean().item())

    return PredictParagraphResponse(
        spiked=spiked,
        spike_class=spike_classes[spike_class_idx],
        spike_values=spike_values,
        mem1=membrane_to_list(mem1),
        mem2=membrane_to_list(mem2),
        mem3=membrane_to_list(mem3),
        membrane_charge=membrane_charge
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(data: PredictRequest):
    """
    End-of-session snapshot classification.
    Uses forward_snapshot (25 internal steps) for a stable final prediction.
    Used for the feedback screen after finishing an article.
    """
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

    actions = ["too_hard", "just_right", "too_easy"]
    return PredictionResponse(
        action=actions[prediction],
        confidence=float(spike_counts.max()),
        raw_scores=spike_counts.tolist()
    )


@app.post("/retrain", response_model=RetrainResponse)
async def retrain(data: RetrainRequest):
    """
    Fine-tunes the model on user samples from localStorage.
    Uses forward_snapshot for training since it gives stable gradients.
    Returns updated weights — frontend saves to localStorage.
    Server stores nothing.
    """
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
    """
    Called when the SNN spikes too_hard mid-article.

    Decides intervention level from genre_difficulty:
      0.5 - 0.7  →  Level 2: full paragraph rewrite in simpler language
      > 0.7      →  Level 3: surgical annotation of only the hard parts

    Calls Claude, returns the altered text and a one-sentence annotation
    explaining what made the paragraph hard. Server stores nothing.
    """
    level = 3 if data.genre_difficulty > 0.7 else 2

    if level == 2:
        prompt = f"""You are a reading assistant helping someone who is struggling with a paragraph.

Rewrite the following paragraph in simpler language. Preserve all the key information and meaning — just make it easier to read. Use shorter sentences, simpler vocabulary, and avoid jargon where possible. Do not add new information or change the facts.

Genre: {data.specific_genre}

Original paragraph:
{data.paragraph}

Respond with a JSON object with exactly two fields:
1. "rewritten": the simplified paragraph as a plain string
2. "annotation": one sentence explaining what made the original paragraph hard (e.g. "This paragraph uses specialized military procurement terminology that assumes prior domain knowledge.")

Respond with ONLY valid JSON, no markdown."""

    else:
        prompt = f"""You are a reading assistant helping someone who is struggling with a dense paragraph.

The paragraph below is from a {data.specific_genre} article with high cognitive density. Rather than rewriting it entirely, identify only the hardest parts — domain-specific terms, dense syntax, or assumed background knowledge — and annotate them inline.

For each hard term or phrase, add a brief clarification in square brackets immediately after it, like this:
"The LIF neurons [leaky integrate-and-fire neurons — a model of how biological neurons accumulate and release electrical charge] process each input..."

Keep the original text exactly as written except for the inline annotations. Do not rewrite sentences, add context before or after, or change the structure.

Original paragraph:
{data.paragraph}

Respond with a JSON object with exactly two fields:
1. "rewritten": the original paragraph with inline annotations added as described
2. "annotation": one sentence identifying the primary source of difficulty (e.g. "This paragraph assumes familiarity with procurement law and military readiness classifications.")

Respond with ONLY valid JSON, no markdown."""

    try:
        message = claude_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = message.content[0].text.strip()
        result = json.loads(raw)

        return InterventionResponse(
            level=level,
            rewritten=result["rewritten"],
            annotation=result.get("annotation", "")
        )

    except Exception as e:
        # Fallback: return original paragraph unchanged so the UI doesn't break
        print(f"Intervention failed: {e}")
        return InterventionResponse(
            level=level,
            rewritten=data.paragraph,
            annotation="Intervention unavailable — showing original paragraph."
        )


@app.post("/ingest-url", response_model=IngestURLResponse)
async def ingest_url(data: IngestURLRequest):
    try:
        downloaded = trafilatura.fetch_url(data.url)
        if not downloaded:
            raise ValueError("Failed to download URL")

        content = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            include_images=False,
            no_fallback=False,
            favor_precision=True,
            favor_recall=False,
            include_formatting=True
        )

        if not content:
            raise ValueError("Failed to extract content from URL")

        metadata = trafilatura.extract_metadata(downloaded)
        title = metadata.title if metadata and metadata.title else "Untitled Article"

        lines = content.split('\n')
        cleaned_paragraphs = []
        skip_patterns = [
            'join war on the rocks', 'subscribe', 'follow', 'share this',
            'image:', 'photo:', 'credit:', 'related articles',
            'recommended reading', 'follow him on twitter', 'follow her on twitter',
            'airman', 'staff sgt', 'tech sgt'
        ]
        current_paragraph = []

        for line in lines:
            line = line.strip()
            if not line:
                if current_paragraph:
                    para_text = ' '.join(current_paragraph)
                    if len(para_text.split()) >= 20:
                        if not any(p in para_text.lower() for p in skip_patterns):
                            cleaned_paragraphs.append(para_text)
                    current_paragraph = []
                continue
            if len(line.split()) < 5:
                continue
            current_paragraph.append(line)

        if current_paragraph:
            para_text = ' '.join(current_paragraph)
            if len(para_text.split()) >= 20:
                if not any(p in para_text.lower() for p in skip_patterns):
                    cleaned_paragraphs.append(para_text)

        clean_content = '\n\n'.join(cleaned_paragraphs)
        word_count = len(clean_content.split())
        paragraph_count = len(cleaned_paragraphs)

        fk_grade = textstat.flesch_kincaid_grade(clean_content)
        estimated_lexile = max(200, min(1600, int(fk_grade * 100 + 200)))

        classification = {}
        try:
            classification = classify_text_with_claude(clean_content)
        except Exception as ce:
            print(f"Classification failed: {ce}")
            classification = {
                "broad_genre": "other", "specific_genre": "unknown",
                "genre_difficulty": 0.5, "reasoning": "Classification unavailable"
            }

        article_id = f"article_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

        return IngestURLResponse(
            status="success",
            title=title,
            content=clean_content,
            estimated_lexile=estimated_lexile,
            word_count=word_count,
            paragraph_count=paragraph_count,
            article_id=article_id,
            classification=classification
        )

    except Exception as e:
        return IngestURLResponse(
            status="error", title="Error", content=str(e),
            estimated_lexile=0, word_count=0, paragraph_count=0,
            article_id="", classification={}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)