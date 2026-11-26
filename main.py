import io
import os
import re
import numpy as np
import pandas as pd
import nltk
from typing import Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# Descarga recursos de NLTK solo si no existen aún en el entorno
NLTK_RESOURCES = [
    ("corpora/stopwords", "stopwords"),
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
]

for resource_path, pkg_name in NLTK_RESOURCES:
    try:
        nltk.data.find(resource_path)
    except LookupError:
        try:
            nltk.download(pkg_name, quiet=True)
        except Exception:
            pass

from nltk.corpus import stopwords

app = FastAPI(title="TextLab API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Almacenamiento en memoria de la sesión actual
SESSION_STATE: Dict[str, Any] = {
    "df": None,
    "filename": None,
    "experiments": {},  # id -> {pipeline, classes, summary}
}

try:
    SPANISH_STOPWORDS = set(stopwords.words("spanish")) | set(stopwords.words("english"))
except LookupError:
    SPANISH_STOPWORDS = set()

TOKEN_RE = re.compile(r"[a-zA-Z0-9áéíóúüñ]+", re.UNICODE)


class LinguisticPreprocessor(BaseEstimator, TransformerMixin):
    """Transformador compatible con Scikit-Learn Pipeline para evitar fuga de datos."""

    def __init__(self, feature_type: str = "ngrams"):
        self.feature_type = feature_type

    def fit(self, X, y=None):
        return self

    def _transform_text(self, text: str) -> str:
        s = str(text).strip().lower()
        if not s:
            return ""

        tokens = TOKEN_RE.findall(s)
        if not tokens:
            return ""

        if self.feature_type == "ngrams":
            return " ".join(tokens)
        elif self.feature_type == "stopwords":
            sw = [t for t in tokens if t in SPANISH_STOPWORDS]
            return " ".join(sw) if sw else "none_sw"
        elif self.feature_type == "content_words":
            cw = [t for t in tokens if t not in SPANISH_STOPWORDS]
            return " ".join(cw) if cw else "none_cw"
        elif self.feature_type == "pos":
            try:
                tagged = nltk.pos_tag(tokens)
                return " ".join([tag for _, tag in tagged]) if tagged else "NONE"
            except Exception:
                return " ".join(tokens)

        return " ".join(tokens)

    def transform(self, X):
        return [self._transform_text(doc) for doc in X]


class DatasetInspectRequest(BaseModel):
    text_col: str
    label_col: str
    drop_duplicates: bool = True
    drop_na: bool = True


class TrainRequest(BaseModel):
    text_col: str
    label_col: str
    feature_type: str             # ngrams, pos, stopwords, content_words
    ngram_size: int               # 1, 2, 3
    ngram_scope: str = "exact"    # exact o up_to
    representation: str           # freq, binary, tfidf
    classifier: str               # logreg, svm, rf, nb
    test_size: float              # e.g. 0.2
    cv_folds: int                 # 0, 3, 5, 10
    params: Dict[str, Any] = {}


class PredictTextRequest(BaseModel):
    experiment_id: str
    text: str


def _read_csv_bytes(content: bytes) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(content), encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.BytesIO(content), encoding="utf-8", errors="replace")


@app.get("/")
def serve_frontend():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    raise HTTPException(status_code=404, detail="index.html no encontrado.")


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.post("/api/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = _read_csv_bytes(content)

        SESSION_STATE["df"] = df
        SESSION_STATE["filename"] = file.filename

        return {
            "status": "ok",
            "filename": file.filename,
            "rows": int(len(df)),
            "columns": list(df.columns),
            "preview": df.head(5).fillna("").to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer CSV: {str(e)}")


@app.post("/api/inspect-dataset")
def inspect_dataset(req: DatasetInspectRequest):
    df = SESSION_STATE.get("df")
    if df is None:
        raise HTTPException(status_code=400, detail="No hay ningún dataset cargado.")

    if req.text_col not in df.columns or req.label_col not in df.columns:
        raise HTTPException(status_code=400, detail="Columnas seleccionadas no válidas.")

    sub = df[[req.text_col, req.label_col]].copy()
    missing_count = int(sub.isna().any(axis=1).sum())
    duplicate_count = int(sub.duplicated().sum())

    if req.drop_na:
        sub = sub.dropna()
    if req.drop_duplicates:
        sub = sub.drop_duplicates()

    sub[req.label_col] = sub[req.label_col].astype(str)
    counts = sub[req.label_col].value_counts()
    total = int(len(sub))

    distribution = [
        {
            "label": str(label),
            "count": int(cnt),
            "percentage": round((int(cnt) / total) * 100, 2) if total > 0 else 0,
        }
        for label, cnt in counts.items()
    ]

    return {
        "total_original": int(len(df)),
        "total_clean": total,
        "missing_rows": missing_count,
        "duplicate_rows": duplicate_count,
        "classes": list(counts.index),
        "distribution": distribution,
    }


@app.post("/api/train")
def train_model(req: TrainRequest):
    df = SESSION_STATE.get("df")
    if df is None:
        raise HTTPException(status_code=400, detail="Carga un dataset primero.")

    if req.text_col not in df.columns or req.label_col not in df.columns:
        raise HTTPException(status_code=400, detail="Columnas seleccionadas no válidas.")

    clean_df = df[[req.text_col, req.label_col]].dropna().drop_duplicates().copy()
    X = clean_df[req.text_col].astype(str).values
    y = clean_df[req.label_col].astype(str).values
    classes = sorted(list(np.unique(y)))

    if len(classes) < 2:
        raise HTTPException(status_code=400, detail="El dataset debe tener al menos 2 clases.")

    n_val = max(1, min(3, int(req.ngram_size)))
    ngram_range = (1, n_val) if req.ngram_scope == "up_to" else (n_val, n_val)
    max_feat = 5000

    if req.representation == "tfidf":
        vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_feat)
    elif req.representation == "binary":
        vectorizer = CountVectorizer(ngram_range=ngram_range, binary=True, max_features=max_feat)
    else:
        vectorizer = CountVectorizer(ngram_range=ngram_range, binary=False, max_features=max_feat)

    p = req.params or {}
    if req.classifier == "logreg":
        penalty = str(p.get("penalty", "l2")).lower()
        c_val = float(p.get("C", 1.0))
        if penalty == "l1":
            clf = LogisticRegression(
                C=c_val,
                l1_ratio=1.0,
                solver="saga",
                max_iter=1000,
                random_state=42,
            )
        else:
            clf = LogisticRegression(
                C=c_val,
                l1_ratio=0.0,
                max_iter=1000,
                random_state=42,
            )
        clf_label = "Regresión Logística"
    elif req.classifier == "svm":
        clf = SVC(
            C=float(p.get("C", 1.0)),
            kernel=p.get("kernel", "linear"),
            probability=False,
            random_state=42,
        )
        clf_label = "SVM"
    elif req.classifier == "rf":
        clf = RandomForestClassifier(
            n_estimators=min(int(p.get("n_estimators", 100)), 150),
            max_depth=int(p["max_depth"]) if p.get("max_depth") else None,
            random_state=42,
            n_jobs=1,
        )
        clf_label = "Random Forest"
    elif req.classifier == "nb":
        clf = MultinomialNB(alpha=float(p.get("alpha", 1.0)))
        clf_label = "Naive Bayes"
    else:
        raise HTTPException(status_code=400, detail="Clasificador no soportado.")

    pipeline = Pipeline([
        ("linguistic", LinguisticPreprocessor(feature_type=req.feature_type)),
        ("vectorizer", vectorizer),
        ("classifier", clf),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=req.test_size, stratify=y, random_state=42
    )

    cv_result = None
    if req.cv_folds in [3, 5, 10]:
        skf = StratifiedKFold(n_splits=req.cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring="accuracy", n_jobs=1)
        cv_result = {
            "folds": req.cv_folds,
            "scores": [round(float(s) * 100, 2) for s in scores],
            "mean": round(float(np.mean(scores)) * 100, 2),
            "std": round(float(np.std(scores)) * 100, 2),
        }

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = round(float(accuracy_score(y_test, y_pred)) * 100, 2)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    prec_cls, rec_cls, f1_cls, sup_cls = precision_recall_fscore_support(
        y_test, y_pred, labels=classes, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=classes).tolist()

    per_class = [
        {
            "label": cls_name,
            "precision": round(float(prec_cls[i]) * 100, 2),
            "recall": round(float(rec_cls[i]) * 100, 2),
            "f1": round(float(f1_cls[i]) * 100, 2),
            "support": int(sup_cls[i]),
        }
        for i, cls_name in enumerate(classes)
    ]

    exp_num = len(SESSION_STATE["experiments"]) + 1
    exp_id = f"EXP-{exp_num:02d}"

    feat_map = {
        "ngrams": f"{n_val}-gramas",
        "pos": f"POS ({n_val}-g)",
        "stopwords": f"Stopwords ({n_val}-g)",
        "content_words": f"Content words ({n_val}-g)",
    }

    result_payload = {
        "id": exp_id,
        "classifier": clf_label,
        "features": f"{feat_map.get(req.feature_type, req.feature_type)} · {req.representation.upper()}",
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "classes": classes,
        "metrics": {
            "accuracy": acc,
            "precision_macro": round(float(prec_macro) * 100, 2),
            "recall_macro": round(float(rec_macro) * 100, 2),
            "f1_macro": round(float(f1_macro) * 100, 2),
        },
        "per_class": per_class,
        "confusion_matrix": cm,
        "cv": cv_result,
    }

    SESSION_STATE["experiments"][exp_id] = {
        "pipeline": pipeline,
        "classes": classes,
        "summary": result_payload,
    }

    return result_payload


@app.post("/api/predict-text")
def predict_text(req: PredictTextRequest):
    exp = SESSION_STATE["experiments"].get(req.experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experimento no encontrado.")

    pipeline = exp["pipeline"]
    classes = exp["classes"]
    pred = str(pipeline.predict([req.text])[0])

    probabilities = []
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba([req.text])[0]
        probabilities = [
            {"label": str(classes[i]), "probability": round(float(probs[i]) * 100, 2)}
            for i in range(len(classes))
        ]
        probabilities.sort(key=lambda x: x["probability"], reverse=True)
    else:
        probabilities = [{"label": pred, "probability": 100.0}]

    return {
        "experiment_id": req.experiment_id,
        "predicted_class": pred,
        "probabilities": probabilities,
    }


@app.post("/api/predict-csv")
async def predict_csv(experiment_id: str, text_col: str, file: UploadFile = File(...)):
    exp = SESSION_STATE["experiments"].get(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experimento no encontrado.")

    content = await file.read()
    df = _read_csv_bytes(content)

    if text_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"La columna '{text_col}' no existe en el CSV.")

    texts = df[text_col].fillna("").astype(str).tolist()
    preds = exp["pipeline"].predict(texts)

    results = []
    has_proba = hasattr(exp["pipeline"], "predict_proba")
    probas = exp["pipeline"].predict_proba(texts) if has_proba else None

    for idx, txt in enumerate(texts):
        conf = round(float(np.max(probas[idx])) * 100, 2) if has_proba else 100.0
        results.append({
            "row": idx + 1,
            "text": txt,
            "predicted_class": str(preds[idx]),
            "confidence": conf,
        })

    return {"total": len(results), "predictions": results}