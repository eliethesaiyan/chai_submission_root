"""Build a portable ZIP with exactly the assessment's allowed root structure."""
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    model = ROOT / "models/entity_extractor"
    for name in ["model.safetensors", "config.json", "tokenizer.json", "metadata.json", "split_manifest.json"]:
        if not (model / name).is_file():
            raise FileNotFoundError(f"Required artifact missing: {model / name}")
    archive = ROOT / "outputs/submission.zip"
    selected = [ROOT / n for n in ["README.md", "Dockerfile", "requirements.txt", "run.sh", ".dockerignore"]]
    # Explicit selection includes weights even though *.safetensors is gitignored.
    for folder in ["src", "models/entity_extractor", "outputs"]:
        for file in (ROOT / folder).rglob("*"):
            relative = file.relative_to(ROOT)
            if folder == "outputs" and file.parent != ROOT / "outputs":
                continue
            if not file.is_file() or "__pycache__" in relative.parts or any(p.startswith("checkpoint-") for p in relative.parts):
                continue
            if file.suffix in {".zip", ".pyc", ".bin", ".log"} or file.name == "package_manifest.json":
                continue
            # Legacy model-side scripts remain in the workspace; canonical code is under src.
            if folder == "models/entity_extractor" and file.suffix == ".py":
                continue
            selected.append(file)
    names = sorted(set(p.relative_to(ROOT).as_posix() for p in selected))
    allowed = {"README.md", "Dockerfile", "requirements.txt", "run.sh", ".dockerignore", "src", "models", "outputs"}
    assert {n.split("/")[0] for n in names} <= allowed
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as bundle:
        for name in names:
            bundle.write(ROOT / name, name)
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        assert "models/entity_extractor/model.safetensors" in bundle.namelist()
    manifest = {"archive": "outputs/submission.zip", "bytes": archive.stat().st_size, "files": names}
    (ROOT / "outputs/package_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Created {archive} ({archive.stat().st_size / 1024**2:.1f} MiB, {len(names)} files)")


if __name__ == "__main__":
    main()
