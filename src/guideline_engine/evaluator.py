"""Deterministic assessment against the supplied fixture, never external advice."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GUIDELINES_PATH = ROOT / "src/data/guidelines.json"


def load_guidelines(path=GUIDELINES_PATH):
    return json.loads(Path(path).read_text())


GUIDELINES = load_guidelines()
GUIDELINE_VERSION = hashlib.sha256(json.dumps(GUIDELINES, sort_keys=True).encode()).hexdigest()[:12]

TEST_ALIASES = {
    "chest_xray": ["chest xray", "chest x-ray", "chest x ray", "cxr"],
    "lumbar_puncture": ["lumbar puncture"],
    "ecg": ["ecg", "ekg", "electrocardiogram"],
    "urinalysis": ["urinalysis", "urine analysis"],
    "sputum_microscopy": ["sputum microscopy"],
    "peak_flow": ["peak flow", "pefr"],
    "fasting_glucose": ["fasting glucose", "fasting blood glucose"],
    "blood_pressure": ["blood pressure", "bp"],
    "malaria_rapid_test": ["malaria rapid test", "malaria rdt"],
    "hemoglobin": ["hemoglobin", "haemoglobin"],
    "echocardiogram": ["echocardiogram", "echocardiography"],
    "spirometry": ["spirometry"], "ct_brain": ["ct brain", "brain ct"],
    "eeg": ["eeg", "electroencephalogram"], "otoscopy": ["otoscopy"],
}


def normalize(value):
    return re.sub(r"\s+", " ", value.strip().lower()) if value is not None else None


def canonical_test(value):
    value = normalize(value).replace("_", " ")
    for name, aliases in TEST_ALIASES.items():
        if value in aliases or value == name.replace("_", " "):
            return name
    return value.replace(" ", "_")


def extract_observed_tests(text):
    """Only affirmative documented mentions count; preserve the exact evidence.

    Conservative sentence-level exclusions mean complex negation needs review.
    'ECG suggests ...' supplies a result and counts; 'needs ECG' does not.
    """
    evidence = []
    for sentence in re.finditer(r"[^.!?;\n]+", text):
        content = sentence.group()
        lower = content.lower()
        excluded = re.search(r"\b(no|not|without|declined|refused|pending|planned|plan|needs?|required|recommend\w*|order\w*|await\w*|consider\w*|defer\w*)\b", lower)
        for test, aliases in TEST_ALIASES.items():
            variants = set(aliases + [test])
            pattern = r"(?<!\w)(?:" + "|".join(re.escape(a) for a in sorted(variants, key=len, reverse=True)) + r")(?!\w)"
            match = re.search(pattern, content, re.I)
            if match and not excluded:
                evidence.append({"test": test, "text": match.group(), "start": sentence.start() + match.start(),
                                 "end": sentence.start() + match.end(), "source": "affirmative_note_mention"})
    return evidence


def evaluate_medications(medications, guideline):
    recommended = {normalize(d) for d in guideline.get("recommended_drugs", [])}
    avoided = {normalize(d) for d in guideline.get("avoid_drugs", [])}
    return [{"medication": d, "status": "avoid" if normalize(d) in avoided else
             "recommended" if normalize(d) in recommended else "not_listed"} for d in medications]


def evaluate_required_tests(observed_tests, guideline):
    observed = {canonical_test(t) for t in observed_tests}
    return [{"test": t, "present": canonical_test(t) in observed} for t in guideline.get("required_tests", [])]


def evaluate_guideline(diagnosis, medications, observed_tests=None, guidelines=None):
    rules = GUIDELINES if guidelines is None else guidelines
    version = GUIDELINE_VERSION if guidelines is None else hashlib.sha256(json.dumps(rules, sort_keys=True).encode()).hexdigest()[:12]
    key = next((name for name in rules if normalize(name) == normalize(diagnosis)), None)
    result = {"status": "evaluated", "diagnosis": diagnosis, "guideline_version": version,
              "compliant": None, "fully_compliant": None, "medications": [], "recommended_medications_given": [],
              "forbidden_medications": [], "recommended_drugs": [], "required_tests": [], "missing_tests": [], "warnings": []}
    if key is None:
        reason = "No diagnosis was extracted." if not diagnosis else "No supplied guideline covers the extracted diagnosis."
        result.update(status="cannot_evaluate" if not diagnosis else "guideline_not_found",
                      reason=reason, explanation=reason, warnings=[reason],
                      medications=[{"medication": d, "status": "unknown"} for d in medications])
        return result
    guideline = rules[key]
    meds = evaluate_medications(medications, guideline)
    tests = evaluate_required_tests(observed_tests or [], guideline)
    recommended = [m["medication"] for m in meds if m["status"] == "recommended"]
    forbidden = [m["medication"] for m in meds if m["status"] == "avoid"]
    unlisted = [m["medication"] for m in meds if m["status"] == "not_listed"]
    missing = [t["test"] for t in tests if not t["present"]]
    warnings = [f"{d} is forbidden by supplied guideline '{key}'." for d in forbidden]
    warnings += [f"{d} is not listed in supplied guideline '{key}'; review required." for d in unlisted]
    if not recommended:
        warnings.append("No recommended medication is documented.")
    warnings += [f"Required test not documented: {t}." for t in missing]
    # The brief's example allows compliant=true alongside missing tests.
    medication_compliant = bool(recommended) and not forbidden and not unlisted
    explanation = (f"Compared documented medications with recommended_drugs and avoid_drugs for '{key}'. "
                   f"Recommended: {', '.join(recommended) or 'none'}; forbidden: {', '.join(forbidden) or 'none'}; "
                   f"unlisted: {', '.join(unlisted) or 'none'}. "
                   f"Required tests not documented: {', '.join(missing) or 'none'}. "
                   "Medication compliance does not establish completeness of a multi-drug regimen.")
    result.update(medications=meds, recommended_medications_given=recommended, forbidden_medications=forbidden,
                  recommended_drugs=guideline.get("recommended_drugs", []), required_tests=tests,
                  missing_tests=missing, warnings=warnings, compliant=medication_compliant,
                  fully_compliant=medication_compliant and not missing, explanation=explanation)
    return result


class GuidelineEvaluator:
    def __init__(self, guidelines_path, predictions_path, output_path):
        self.guidelines_path = Path(guidelines_path)
        self.predictions_path = Path(predictions_path)
        self.output_path = Path(output_path)
        self.guidelines = load_guidelines(self.guidelines_path)

    def evaluate(self, diagnosis, medications, observed_tests=None):
        return evaluate_guideline(diagnosis, medications, observed_tests, self.guidelines)

    def save(self, result):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(result, indent=2) + "\n")

    def run(self):
        predictions = json.loads(self.predictions_path.read_text())
        def assess(p):
            structured = p.get("structured", p)
            tests = p.get("observed_tests", [e["test"] for e in extract_observed_tests(p.get("text", ""))])
            return {"note_id": p.get("note_id", ""), **self.evaluate(structured.get("diagnosis"), structured.get("medications", []), tests)}
        result = [assess(p) for p in predictions] if isinstance(predictions, list) else assess(predictions)
        self.save(result)
        return result
