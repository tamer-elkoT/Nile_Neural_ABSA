# Project Documentation: Arabic Aspect-Based Sentiment Analysis (ABSA)

## 1. Project Overview

This project aims to build a production-ready Arabic ABSA system that analyzes customer reviews from multiple domains (restaurants, delivery apps, healthcare, retail) and identifies:

- **Aspects** mentioned in each review (from a fixed taxonomy of 9 classes)
- **Sentiment** per aspect (positive, negative, neutral)

The output must be a JSON file in the format:

```json
[
  {
    "review_id": 23,
    "aspects": ["service", "ambiance", "food"],
    "aspect_sentiments": {
      "service": "positive",
      "ambiance": "positive",
      "food": "negative"
    }
  }
]
```

The evaluation metric is **micro F1** over (aspect, sentiment) pairs.  
We will follow a two‑phase approach:

- **Phase A (Baseline)**: Fine‑tuned AraBERT pipeline (aspect detection + per‑aspect sentiment classifiers).
- **Phase B (Advanced)**: SetFit‑based Arabic ABSA (data‑efficient, using sentence transformers).

The project is split equally among **3 team members**. Each member has defined responsibilities, deliverables, and a tight **one‑day timeline** (1:00 PM – 8:00 PM today).

---

## 2. Team Member Assignments

| Member | Primary Role | Key Responsibilities |
|--------|--------------|----------------------|
| **Member A** | Data & Baseline Models | Data preprocessing, aspect detection model, per‑aspect sentiment classifiers, baseline inference |
| **Member B** | SetFit & Advanced Pipeline | Candidate extraction (Arabic), SetFit training for aspect filtering and sentiment, advanced inference |
| **Member C** | Evaluation, Integration & Submission | Metric implementation, model comparison, final submission JSON, documentation & presentation |

All members collaborate on code review, debugging, and integration.

---

## 3. Phase 1 – Data Preparation & Exploration (All members, but Member A leads)

### 3.1 Tasks

1. **Load and inspect** the three Excel files (`train_fixed.xlsx`, `validation_fixed.xlsx`, `unlabeled_fixed.xlsx`).  
   - Print basic statistics: number of reviews, aspect frequencies, sentiment distributions.  
   - Sample random reviews to understand language patterns.

2. **Implement a unified preprocessing function** `preprocess(text)` that:  
   - Normalises Arabic letters (unify Alef, remove diacritics).  
   - Replaces emojis with descriptive tokens (e.g., `😍` → `_love_`).  
   - Removes URLs, mentions, extra whitespace.  
   - Keeps punctuation (exclamation marks, question marks).  
   - Optionally lowercases (Arabic transformers are case‑sensitive? Usually not needed).

3. **Create aspect detection datasets**  
   - For each review, convert the `aspects` list into a 9‑dimensional binary vector.  
   - Save `X_train_aspect` (preprocessed texts) and `y_train_aspect` (binary arrays).  
   - Do the same for validation data.

4. **Create sentiment datasets per aspect**  
   - For each of the 9 aspects, collect reviews where that aspect appears.  
   - Extract sentiment labels from `aspect_sentiments`.  
   - Save as `X_train_sent_{aspect}` and `y_train_sent_{aspect}`.

**Deliverables (Member A):**  
- Cleaned datasets (`.npy` or `.pkl` files).  
- Preprocessing function saved as `preprocess.py`.  
- Data statistics report (Markdown or notebook).

---

## 4. Phase 2 – Baseline Pipeline (AraBERT) – Member A

### 4.1 Architecture

- **Aspect detection**: `aubmindlab/bert-base-arabertv2` with 9‑output sigmoid head.  
  Loss: Binary Cross Entropy.

- **Per‑aspect sentiment classifiers**: Same BERT with 3‑output softmax head.  
  Train one model per aspect (9 models).  
  Use class weights to handle imbalance.

### 4.2 Implementation Steps

1. **Install dependencies**: `transformers`, `torch`, `datasets`, `scikit-learn`, `pandas`.  
2. **Load pre‑trained model** and tokenizer.  
3. **Write training loop** using Hugging Face `Trainer` or custom PyTorch.  
   - Use `AdamW` optimizer, linear warmup.  
   - Monitor validation loss and F1.  
4. **Aspect detection training:**  
   - Train for 3‑5 epochs, batch size 16.  
   - Save best model based on validation micro F1.  
   - Tune threshold per aspect (0.3–0.7) on validation set.  
5. **Sentiment classifier training (for each aspect):**  
   - Extract subset of reviews where aspect appears.  
   - Train for 3‑5 epochs, using class weights.  
   - Save each model as `sentiment_{aspect}.bin`.  
6. **Combine into inference function** `predict_baseline(text)` that returns `(aspects_list, sentiments_dict)`.  
7. **Run on validation set** and compute micro F1 score (use evaluation script from Member C).  
8. **Generate baseline submission** for the unlabeled test set.

**Deliverables (Member A):**  
- Trained aspect detection model.  
- 9 trained sentiment classifiers.  
- Baseline inference pipeline (`baseline_pipeline.py`).  
- Baseline submission JSON (`submission_baseline.json`).  
- Validation performance report (micro F1, per‑aspect scores).

---

## 5. Phase 3 – Advanced Pipeline (SetFit‑based) – Member B

### 5.1 Rationale

SetFit (Sentence Transformer Fine‑tuning) is data‑efficient and works well with limited labeled data. We implement a custom three‑stage ABSA pipeline for Arabic:

1. **Candidate extraction** – Find possible aspect phrases in the review.  
2. **Aspect filtering** – Binary SetFit classifier to decide if a candidate is a valid aspect.  
3. **Sentiment polarity** – 3‑class SetFit classifier for each aspect.

### 5.2 Implementation Steps

#### 5.2.1 Candidate Extraction (Arabic)

Because spaCy’s English ABSA extension is not Arabic‑compatible, implement a custom extractor:

- **Option 1 (simple):** Use a keyword list from the training data – collect all unique aspect tokens that appear as spans (or exact matches). Then use a fuzzy string matcher (e.g., `rapidfuzz`) to find occurrences in new reviews.  
- **Option 2 (more robust):** Use `stanza` for Arabic (Stanford NLP) to extract noun phrases (NP) from the constituency parse.  
  ```python
  import stanza
  stanza.download('ar')
  nlp = stanza.Pipeline('ar', processors='tokenize,pos,constituency')
  doc = nlp(review)
  # Extract NP constituents
  ```
- **Option 3 (fallback):** Use `spacy` with a blank Arabic pipeline + a simple regex to extract sequences of words that match Arabic noun patterns (e.g., `ال...`). This is less accurate but faster.

**Implement at least two methods** and choose the one with best recall on validation.

#### 5.2.2 Build SetFit Training Data

- **For aspect filtering:**  
  - Positive examples: `(review_text, candidate_span)` where the candidate appears in the review AND is listed in the review’s `aspects`.  
  - Negative examples: `(review_text, candidate_span)` where the candidate is a noun phrase (or random word) from the review that is NOT in the review’s `aspects`.

- **For sentiment polarity (per aspect):**  
  - For each aspect `a`, collect `(review_text, candidate_span)` where the candidate matches exactly the aspect phrase in the review, and label is taken from `aspect_sentiments[a]`.

Use the multilingual sentence transformer `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (good for Arabic).  
Optionally try Arabic‑specific ones like `sentence-transformers/paraphrase-MiniLM-L12-ar-v1`.

#### 5.2.3 Train SetFit Models

Use the `setfit` library:

```python
from setfit import SetFitModel, Trainer, TrainingArguments

model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
args = TrainingArguments(
    batch_size=16,
    num_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    output_dir="./setfit_filter",
)
trainer = Trainer(model, args, train_dataset=filter_train, eval_dataset=filter_val)
trainer.train()
```

Train:
- One binary classifier for aspect filtering (shared across all aspects).  
- One 3‑class classifier **per aspect** for sentiment (9 models).

#### 5.2.4 Inference Pipeline

For a new review:

1. Run candidate extraction → list of candidate spans.  
2. For each candidate, use the filtering model to predict `is_aspect`. If yes, keep it.  
3. For each kept aspect, use the corresponding sentiment model to get polarity.  
4. Return aspects and sentiments.

**Optimisation:**  
- Remove duplicate spans (same text).  
- Use a confidence threshold for filtering (e.g., >0.5).

#### 5.2.5 Evaluation and Submission

- Run on validation set, compute micro F1 using Member C’s script.  
- Compare with baseline.  
- Generate `submission_setfit.json` for test set.

**Deliverables (Member B):**  
- Candidate extractor module (`candidate_extractor.py`).  
- Trained SetFit models (filter + 9 sentiment).  
- SetFit inference pipeline (`setfit_pipeline.py`).  
- Validation performance report.  
- Advanced submission JSON (`submission_setfit.json`).

---

## 6. Phase 4 – Evaluation, Integration & Final Submission – Member C

### 6.1 Metric Implementation

Write a function `compute_metrics(true_file, pred_file)` that:

- Loads true labels (from `validation_fixed.xlsx`) and predictions (JSON).  
- Flatten true and predicted into lists of tuples `(review_id, aspect, sentiment)`.  
- Use `sklearn.metrics.f1_score` with `average='micro'`.  
- Also compute per‑aspect F1 for debugging.

### 6.2 Model Comparison

- Run both baseline and SetFit pipelines on the validation set.  
- Produce a table comparing:
  - Overall micro F1.  
  - Per‑aspect F1.  
  - Inference time per review (estimate).  
  - Model size (disk).

### 6.3 Final Model Selection

- If SetFit performs better (or similar but more robust to rare aspects), select it as the final model.  
- Otherwise, select baseline.  
- Document the decision.

### 6.4 Generate Final Submission

- Run the selected pipeline on `unlabeled_fixed.xlsx`.  
- Ensure output JSON matches the exact schema (no extra fields).  
- Validate that all `review_id`s are present and unique.  
- Save as `submission_final.json`.

### 6.5 Documentation and Presentation

- Write a comprehensive markdown report (this document plus results).  
- Include:
  - Data exploration insights (charts if possible).  
  - Model architecture diagrams.  
  - Training hyperparameters.  
  - Validation scores and analysis.  
  - Sample predictions (good and bad cases).  
  - Lessons learned and future improvements.  
- Prepare a 10‑minute presentation (slides) summarising the project.

**Deliverables (Member C):**  
- Evaluation script (`evaluate.py`).  
- Comparison table (Markdown).  
- Final submission JSON (`submission_final.json`).  
- Project report (`report.md`).  
- Presentation slides (`presentation.pdf` or `.pptx`).

---

## 7. Timeline (Today: 1:00 PM – 8:00 PM)

| Time Slot | Member A | Member B | Member C |
|-----------|----------|----------|----------|
| **1:00 – 2:00** | Load data, implement preprocessing, create aspect detection datasets | Research candidate extraction methods, install `stanza`/`spacy` | Implement evaluation metric stub, load validation data |
| **2:00 – 3:30** | Train aspect detection model, tune thresholds | Implement candidate extractor (keyword + stanza), build SetFit training data | Write full `compute_metrics` script, test on dummy predictions |
| **3:30 – 5:00** | Train 9 sentiment classifiers (parallel if possible) | Train SetFit filter model + 9 sentiment models | Prepare comparison framework, run baseline evaluation (as soon as ready) |
| **5:00 – 6:00** | Build baseline inference pipeline, generate `submission_baseline.json` | Build SetFit inference pipeline, generate `submission_setfit.json` | Run SetFit evaluation, produce comparison table |
| **6:00 – 7:00** | Assist Member C with final selection, fix any bugs | Optimise candidate extraction (improve recall), assist Member C | Decide final model, generate `submission_final.json`, start writing report |
| **7:00 – 8:00** | Review report, prepare to present | Review report, prepare to present | Finalise report, create slides, push all deliverables to GitHub |

**Note:** All code must be pushed to a shared GitHub repository with clear commit messages. Each member must test their module independently before integration.

---

## 8. Deployment Process

After generating the final submission JSON, the solution must be deployable as a lightweight API for production use.

### 8.1 Local Deployment (FastAPI + Docker)

We will package the selected model (baseline or SetFit) into a FastAPI application.

**Steps:**

1. **Create a `Dockerfile`**:
   ```dockerfile
   FROM python:3.10-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY . .
   CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

2. **Write `api.py`** with two endpoints:
   - `POST /predict` – accepts a review text, returns aspects and sentiments.
   - `GET /health` – health check.

   Example:
   ```python
   from fastapi import FastAPI
   from pydantic import BaseModel
   from baseline_pipeline import predict_baseline  # or setfit_pipeline

   app = FastAPI()

   class ReviewRequest(BaseModel):
       text: str

   @app.post("/predict")
   def predict(request: ReviewRequest):
       aspects, sentiments = predict_baseline(request.text)
       return {"aspects": aspects, "aspect_sentiments": sentiments}
   ```

3. **Build and run the container**:
   ```bash
   docker build -t absa-arabic .
   docker run -p 8000:8000 absa-arabic
   ```

4. **Test the API**:
   ```bash
   curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"text": "الاكل كان لذيذ لكن الخدمة سيئة"}'
   ```

### 8.2 Cloud Deployment (Optional)

- **Hugging Face Spaces** (free GPU): Upload the model and FastAPI app to a Space with Docker.  
- **AWS/GCP/Azure**: Deploy the container to ECS/GKE/ACI with auto‑scaling.

### 8.3 Deployment Deliverables (All members together)

- `Dockerfile` and `requirements.txt` in the repository root.  
- `api.py` with FastAPI endpoints.  
- Instructions in `README.md` on how to build and run the container locally.  
- (Optional) Link to a deployed endpoint (e.g., Hugging Face Space).

---

## 9. Required Libraries

Install the following using `pip`:

```bash
# Core
pandas, numpy, scikit-learn, matplotlib, seaborn

# Transformers
torch, transformers, datasets

# Arabic preprocessing
pyarabic, stanza, spacy  # (spacy with blank model)

# SetFit
setfit, sentence-transformers

# Deployment
fastapi, uvicorn, docker

# Evaluation & utilities
tqdm, json, re
```

---

## 10. Success Criteria

- [ ] All three members complete their tasks on time (by 8:00 PM today).  
- [ ] Baseline model achieves micro F1 > 0.70 on validation set (estimated).  
- [ ] SetFit model matches or exceeds baseline performance.  
- [ ] Final submission is in correct JSON format and passes schema validation.  
- [ ] Report clearly explains methodology, results, and decisions.  
- [ ] Code is well‑structured, commented, and reproducible.  
- [ ] Docker container runs successfully and returns predictions via API.

---

## 11. Appendix – Aspect Labels and Order

Use the following fixed order for binary vectors:

```python
ASPECTS = [
    "food",
    "service",
    "price",
    "cleanliness",
    "delivery",
    "ambiance",
    "app_experience",
    "general",
    "none"
]
```

Sentiment values: `"positive"`, `"negative"`, `"neutral"`.

---

**End of Documentation**