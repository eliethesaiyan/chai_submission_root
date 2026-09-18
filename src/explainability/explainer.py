"""Gradient × input feature attribution plus traceable guideline evidence."""
from __future__ import annotations
import json
from pathlib import Path
import torch

from src.extraction.predictor import EntityPredictor
from src.guideline_engine.evaluator import evaluate_guideline, extract_observed_tests

ROOT = Path(__file__).resolve().parents[2]


def explain_prediction(predictor, prediction, assessment):
    text = prediction["text"]
    candidates = [e for e in prediction["entities"] if e["label"] == "diagnosis"
                  and e["assertion"] not in {"negated", "historical_or_excluded"}]
    attribution = []
    target = candidates[0] if candidates else None
    if target:
        encoding, offsets = predictor.encode(text)
        model = predictor.model
        # Only embeddings require gradients; parameter .grad fields remain untouched.
        embeddings = model.get_input_embeddings()(encoding["input_ids"]).detach().requires_grad_(True)
        logits = model(inputs_embeds=embeddings, attention_mask=encoding["attention_mask"]).logits[0]
        log_probs = logits.log_softmax(dim=-1)
        positions = [i for i, (s, e) in enumerate(offsets) if s < target["end"] and e > target["start"] and s != e]
        if positions:
            # Explain the actual predicted BIO labels, including an orphan I-tag
            # repaired by span reconstruction; do not substitute a different B-tag.
            labels = logits[positions].detach().argmax(dim=-1).tolist()
            objective = torch.stack([log_probs[p, label] for p, label in zip(positions, labels)]).mean()
            gradient = torch.autograd.grad(objective, embeddings)[0]
            signed = (gradient * embeddings).sum(dim=-1)[0].detach().cpu()
            denominator = sum(abs(float(signed[i])) for i, (s, e) in enumerate(offsets) if s != e) or 1.0
            attribution = [{"text": text[s:e], "start": s, "end": e,
                            "importance": abs(float(signed[i])) / denominator,
                            "signed_score": float(signed[i]) / denominator}
                           for i, (s, e) in enumerate(offsets) if s != e]
    return {"method": "gradient_x_input", "target": target,
            "token_importance": attribution,
            "interpretation": "Importance is normalized absolute gradient × embedding for mean diagnosis BIO log-probability. It measures local model sensitivity, not causation or clinical correctness.",
            "entity_evidence": prediction["entities"],
            "medication_reasons": [{"medication": item["medication"], "status": item["status"],
                                    "reason": f"Exact normalized comparison with supplied recommended_drugs/avoid_drugs: {item['status']}."}
                                   for item in assessment["medications"]],
            "guideline_reason": assessment["explanation"],
            "test_evidence": extract_observed_tests(text),
            "limitations": ["Small weakly labeled training set", "Assertion rules are conservative heuristics",
                           "Scores are not calibrated", "Attributions may reflect dataset shortcuts"]}


def main():
    predictor = EntityPredictor()
    note = json.loads((ROOT / "src/data/clinical_notes.json").read_text())[0]
    prediction = predictor.predict(note)
    tests = [e["test"] for e in extract_observed_tests(note["text"])]
    assessment = evaluate_guideline(prediction["structured"]["diagnosis"], prediction["structured"]["medications"], tests)
    output = explain_prediction(predictor, prediction, assessment)
    (ROOT / "outputs/explanation_example.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"target": output["target"], "top_tokens": sorted(output["token_importance"], key=lambda t: -t["importance"])[:8]}, indent=2))


if __name__ == "__main__":
    main()
