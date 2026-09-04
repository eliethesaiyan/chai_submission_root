import json
from pathlib import Path


GUIDELINES_PATH = Path(__file__).resolve().parent.parent.parent / "src/data/guidelines.json"
print(GUIDELINES_PATH)


def load_guidelines(
    path=GUIDELINES_PATH,
):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


GUIDELINES = load_guidelines()


def normalize(value):
    if value is None:
        return None

    return value.strip().lower()


def evaluate_medications(
    medications,
    guideline,
):
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
    observed_tests,
    guideline,
):
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
    diagnosis,
    medications,
    observed_tests=None,
):
    if observed_tests is None:
        observed_tests = []

    normalized_diagnosis = normalize(
        diagnosis
    )

    if not normalized_diagnosis:
        return {
            "status": "cannot_evaluate",
            "reason": "No diagnosis was extracted.",
        }

    guideline = GUIDELINES.get(
        normalized_diagnosis
    )

    if guideline is None:
        return {
            "status": "guideline_not_found",
            "diagnosis": diagnosis,
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
