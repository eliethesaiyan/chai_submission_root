import json
from pathlib import Path
from typing import Any


GUIDELINES_PATH = Path(__file__).resolve().parents[2] / "src/data/guidelines.json"


def load_guidelines(
    path: str | Path = GUIDELINES_PATH,
) -> dict[str, dict[str, list[str]]]:
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


GUIDELINES = load_guidelines()


def normalize(value: str | None) -> str | None:
    if value is None:
        return None

    return value.strip().lower()


def evaluate_medications(
    medications: list[str],
    guideline: dict[str, Any],
) -> list[dict[str, str]]:
    recommended = {
        normalize(drug)
        for drug in guideline.get(
            "recommended_drugs",
            [],
        )
    }

    avoided = {
        normalize(drug)
        for drug in guideline.get(
            "avoid_drugs",
            [],
        )
    }

    medication_results = []

    for medication in medications:
        normalized_medication = normalize(
            medication
        )

        if normalized_medication in avoided:
            status = "avoid"

        elif normalized_medication in recommended:
            status = "recommended"

        else:
            status = "not_listed"

        medication_results.append(
            {
                "medication": medication,
                "status": status,
            }
        )

    return medication_results


def evaluate_required_tests(
    observed_tests: list[str],
    guideline: dict[str, Any],
) -> list[dict[str, str | bool]]:
    required_tests = guideline.get(
        "required_tests",
        [],
    )

    observed_normalized = {
        normalize(test)
        for test in observed_tests
    }

    results = []

    for test in required_tests:
        normalized_test = normalize(
            test
        )

        present = (
            normalized_test
            in observed_normalized
        )

        results.append(
            {
                "test": test,
                "present": present,
            }
        )

    return results


def evaluate_guideline(
    diagnosis: str | None,
    medications: list[str],
    observed_tests: list[str] | None = None,
) -> dict:
    if observed_tests is None:
        observed_tests = []

    normalized_diagnosis = normalize(
        diagnosis
    )

    if not normalized_diagnosis:
        return {
            "status": "cannot_evaluate",
            "diagnosis": diagnosis,
            "medications": [
                {
                    "medication": medication,
                    "status": "not_listed",
                }
                for medication in medications
            ],
            "recommended_medications_given": [],
            "forbidden_medications": [],
            "recommended_drugs": [],
            "required_tests": [],
            "missing_tests": [],
            "reason": "No diagnosis was extracted.",
        }

    guideline = next(
        (
            value
            for name, value in GUIDELINES.items()
            if normalize(name) == normalized_diagnosis
        ),
        None,
    )

    if guideline is None:
        return {
            "status": "guideline_not_found",
            "diagnosis": diagnosis,
            "medications": [
                {
                    "medication": medication,
                    "status": "not_listed",
                }
                for medication in medications
            ],
            "recommended_medications_given": [],
            "forbidden_medications": [],
            "recommended_drugs": [],
            "required_tests": [],
            "missing_tests": [],
            "reason": (
                "No guideline is available "
                "for the extracted diagnosis."
            ),
        }

    medication_results = (
        evaluate_medications(
            medications,
            guideline,
        )
    )

    test_results = (
        evaluate_required_tests(
            observed_tests,
            guideline,
        )
    )

    recommended_drugs = guideline.get(
        "recommended_drugs",
        [],
    )

    missing_tests = [
        result["test"]
        for result in test_results
        if not result["present"]
    ]

    forbidden_medications = [
        result["medication"]
        for result in medication_results
        if result["status"] == "avoid"
    ]

    recommended_medications_given = [
        result["medication"]
        for result in medication_results
        if result["status"] == "recommended"
    ]

    return {
        "status": "evaluated",
        "diagnosis": diagnosis,

        "medications": medication_results,

        "recommended_medications_given":
            recommended_medications_given,

        "forbidden_medications":
            forbidden_medications,

        "recommended_drugs":
            recommended_drugs,

        "required_tests":
            test_results,

        "missing_tests":
            missing_tests,
    }


class GuidelineEvaluator:
    """Configurable wrapper around the deterministic guideline decision logic."""

    def __init__(
        self,
        guidelines_path: str | Path,
        predictions_path: str | Path,
        output_path: str | Path,
    ) -> None:
        self.guidelines_path = Path(guidelines_path)
        self.predictions_path = Path(predictions_path)
        self.output_path = Path(output_path)

        if self.guidelines_path.resolve() != GUIDELINES_PATH.resolve():
            raise ValueError(
                "The guideline engine must use src/data/guidelines.json; "
                f"received {self.guidelines_path.resolve()}"
            )

    def evaluate(
        self,
        diagnosis: str | None,
        medications: list[str],
        observed_tests: list[str] | None = None,
    ) -> dict:
        return evaluate_guideline(
            diagnosis=diagnosis,
            medications=medications,
            observed_tests=observed_tests,
        )

    def save(self, result: dict) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as file:
            json.dump(result, file, indent=2, ensure_ascii=False)

    def run(self) -> dict:
        """Evaluate the structured entity-extraction output and save the result."""
        with self.predictions_path.open("r", encoding="utf-8") as file:
            prediction = json.load(file)
        if not isinstance(prediction, dict):
            raise ValueError(
                f"Expected one structured prediction object in {self.predictions_path}"
            )

        result = self.evaluate(
            diagnosis=prediction.get("diagnosis"),
            medications=prediction.get("medications", []),
            observed_tests=prediction.get("observed_tests"),
        )
        self.save(result)
        return result


def main():
    example = evaluate_guideline(
        diagnosis="urinary tract infection",
        medications=["ciprofloxacin"],
        observed_tests=[
            "chest_xray",
        ],
    )

    print(
        json.dumps(
            example,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
