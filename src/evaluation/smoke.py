"""Save real API response and privacy-preserving log examples from supplied notes."""
import json
import logging
from pathlib import Path
from fastapi.testclient import TestClient
from src.api.app import create_app

ROOT = Path(__file__).resolve().parents[2]


def main():
    events = []
    class Capture(logging.Handler):
        def emit(self, record):
            events.append(json.loads(record.getMessage()))
    logger = logging.getLogger("clinical_ai.audit")
    handler = Capture()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        notes = json.loads((ROOT / "src/data/clinical_notes.json").read_text())
        with TestClient(create_app()) as client:
            assert client.get("/health").status_code == 200
            responses = []
            for note in [notes[0], notes[3], notes[20]]:
                response = client.post("/analyze_note", json={"text": note["text"], "note_id": note["note_id"]})
                response.raise_for_status()
                responses.append(response.json())
        (ROOT / "outputs/api_examples.json").write_text(json.dumps(responses, indent=2) + "\n")
        (ROOT / "outputs/prediction_log_examples.json").write_text(json.dumps(events, indent=2) + "\n")
        print(f"Saved {len(responses)} API examples, including a forbidden medication and a documented ECG.")
    finally:
        logger.removeHandler(handler)


if __name__ == "__main__":
    main()
