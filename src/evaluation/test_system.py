"""Behavioral regression tests and real-model/API integration checks."""
import json
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.evaluation.evaluate import EntityEvaluator
from src.extraction.predictor import EntityPredictor
from src.guideline_engine.evaluator import evaluate_guideline, extract_observed_tests
from src.preprocessing.annotator import Annotator
from src.training.train import EntityTrainer

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as value:
        yield value


def test_pneumonia_fixture_semantics():
    result = evaluate_guideline(" Pneumonia ", ["AMOXICILLIN"])
    assert result["compliant"] is True
    assert result["fully_compliant"] is False
    assert result["missing_tests"] == ["chest_xray"]
    assert "not documented" in result["warnings"][0]


def test_forbidden_takes_precedence():
    result = evaluate_guideline("pneumonia", ["amoxicillin", "ciprofloxacin"], ["CXR"])
    assert result["compliant"] is False
    assert result["forbidden_medications"] == ["ciprofloxacin"]
    assert result["missing_tests"] == []


@pytest.mark.parametrize("diagnosis,status", [(None, "cannot_evaluate"), ("unknown disease", "guideline_not_found")])
def test_unknown_is_not_compliant(diagnosis, status):
    result = evaluate_guideline(diagnosis, ["amoxicillin"])
    assert result["status"] == status
    assert result["compliant"] is None
    assert result["medications"][0]["status"] == "unknown"


def test_no_medications_and_unlisted():
    assert evaluate_guideline("pneumonia", [])["compliant"] is False
    assert evaluate_guideline("pneumonia", ["unlisted"])["medications"][0]["status"] == "not_listed"


def test_custom_guidelines_are_isolated():
    rules = {"example": {"recommended_drugs": ["x"], "avoid_drugs": ["x"], "required_tests": []}}
    assert evaluate_guideline("example", ["x"], guidelines=rules)["medications"][0]["status"] == "avoid"
    assert evaluate_guideline("example", ["x"])["status"] == "guideline_not_found"


@pytest.mark.parametrize("text,expected", [
    ("ECG suggests myocardial infarction.", ["ecg"]),
    ("Chest X-ray normal.", ["chest_xray"]),
    ("No ECG performed.", []),
    ("ECG not done.", []),
    ("Needs chest xray.", []),
    ("Urinalysis pending.", []),
    ("Plan lumbar puncture.", []),
    ("Ordered chest x-ray.", []),
    ("No ECG. Urinalysis normal.", ["urinalysis"]),
])
def test_test_evidence(text, expected):
    evidence = extract_observed_tests(text)
    assert [e["test"] for e in evidence] == expected
    for e in evidence:
        assert text[e["start"]:e["end"]] == e["text"]


def test_annotation_ids_and_absent_demographics():
    a = Annotator(ROOT / "src/data", ROOT / "src/data/clinical_notes.json",
                  ROOT / "src/data/guidelines.json", ROOT / "src/data/annotations.json")
    result = a.annotate_entities_in_text({"note_id": "X", "text": "Diagnosed pneumonia."},
                                         a.build_diagnostic_vocabulary(), a.build_medication_vocabulary())
    assert result["note_id"] == "X"
    assert all(e is not None for e in result["entities"])
    assert [e["label"] for e in result["entities"]] == ["diagnosis"]


def test_split_ids_and_templates_disjoint():
    manifest = json.loads((ROOT / "models/entity_extractor/split_manifest.json").read_text())
    notes = json.loads((ROOT / "src/data/annotations.json").read_text())
    groups = [{EntityTrainer.template_group(n) for n in notes if n["note_id"] in manifest[split]}
              for split in ["train", "validation", "test"]]
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    assert len(set(manifest["train"] + manifest["validation"] + manifest["test"])) == len(notes)


def test_evaluation_joins_by_id_and_rejects_missing():
    a = {"note_id": "a", "text": "fever", "entities": [{"label": "symptoms", "text": "fever", "start": 0, "end": 5}]}
    b = {"note_id": "b", "text": "cough", "entities": [{"label": "symptoms", "text": "cough", "start": 0, "end": 5}]}
    assert EntityEvaluator.score_notes([a, b], [b, a])["overall"]["f1"] == 1
    with pytest.raises(ValueError):
        EntityEvaluator.score_notes([a, b], [a])
    with pytest.raises(ValueError):
        EntityEvaluator.score_notes([a, a], [a, a])


def test_strict_span_boundaries():
    gold = {"note_id": "a", "text": "chest pain", "entities": [{"label": "symptoms", "start": 0, "end": 10}]}
    prediction = {**gold, "entities": [{"label": "symptoms", "start": 6, "end": 10}]}
    result = EntityEvaluator.score_notes([gold], [prediction])["overall"]
    assert result["true_positives"] == 0
    assert result["false_positives"] == result["false_negatives"] == 1


@pytest.mark.parametrize("text,word,expected", [
    ("No pneumonia.", "pneumonia", "negated"),
    ("History of pneumonia.", "pneumonia", "historical_or_excluded"),
    ("Suspected pneumonia.", "pneumonia", "uncertain"),
    ("Diagnosed pneumonia.", "pneumonia", "affirmed"),
    ("Ciprofloxacin was not given.", "Ciprofloxacin", "negated"),
])
def test_assertion_postprocessor(text, word, expected):
    start = text.index(word)
    assert EntityPredictor.assertion(text, {"start": start, "end": start + len(word)}) == expected


def test_health(client):
    result = client.get("/health")
    assert result.status_code == 200
    assert result.json()["model_version"].startswith("distilbert")


@pytest.mark.parametrize("body", [{}, {"text": ""}, {"text": "  "}, {"text": 3},
                                  {"text": "x", "unknown": True}, {"text": "x" * 12001},
                                  {"text": "x", "observed_tests": [3]}])
def test_api_validation(client, body):
    assert client.post("/analyze_note", json=body).status_code == 422


def test_no_silent_truncation(client):
    result = client.post("/analyze_note", json={"text": "fever " * 600})
    assert result.status_code == 422
    assert "token limit" in result.json()["detail"]


def test_real_model_api_attribution_and_private_logs(client, caplog):
    text = json.loads((ROOT / "src/data/clinical_notes.json").read_text())[0]["text"]
    with caplog.at_level(logging.INFO, logger="clinical_ai.audit"):
        result = client.post("/analyze_note", json={"text": text, "note_id": "private-id", "observed_tests": ["CXR"]})
    assert result.status_code == 200, result.text
    data = result.json()
    assert data["structured_entities"] == {"note_id": "private-id", "age": 45, "sex": "male",
                                           "symptoms": ["fever", "productive cough"], "diagnosis": "pneumonia",
                                           "medications": ["amoxicillin"]}
    assert data["guideline_assessment"]["fully_compliant"] is True
    importance = data["explanation"]["token_importance"]
    assert importance and sum(t["importance"] for t in importance) == pytest.approx(1)
    assert all(text[t["start"]:t["end"]] == t["text"] for t in importance)
    assert data["latency_ms"] > 0
    events = [json.loads(r.message) for r in caplog.records if r.name == "clinical_ai.audit"]
    assert events[-1]["event"] == "prediction"
    assert text not in caplog.text and "private-id" not in caplog.text and "amoxicillin" not in caplog.text


def test_forbidden_medication_end_to_end(client):
    notes = json.loads((ROOT / "src/data/clinical_notes.json").read_text())
    result = client.post("/analyze_note", json={"text": notes[3]["text"]}).json()
    assert result["guideline_assessment"]["forbidden_medications"] == ["ciprofloxacin"]
    assert result["guideline_assessment"]["compliant"] is False


def test_model_missing_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing model artifacts"):
        EntityPredictor(model_dir=tmp_path).load()


def test_saved_api_examples():
    from src.evaluation.smoke import main
    main()
    examples = json.loads((ROOT / "outputs/api_examples.json").read_text())
    events = json.loads((ROOT / "outputs/prediction_log_examples.json").read_text())
    assert len(examples) == 3
    assert examples[1]["guideline_assessment"]["forbidden_medications"] == ["ciprofloxacin"]
    assert "ecg" not in examples[2]["guideline_assessment"]["missing_tests"]
    assert sum(e["event"] == "prediction" for e in events) == 3
    assert all("text" not in e for e in events)
