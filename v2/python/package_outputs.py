from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def add_tree(zip_file: ZipFile, root: Path, base: Path) -> None:
    if not root.exists():
        return
    if root.is_file():
        zip_file.write(root, root.relative_to(base))
        return
    for path in root.rglob("*"):
        if path.is_file():
            zip_file.write(path, path.relative_to(base))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(".."))
    parser.add_argument("--output", type=Path, default=Path("../deliverables_rtl_bundle.zip"))
    args = parser.parse_args()

    base = args.project_root.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    with ZipFile(output, "w", compression=ZIP_DEFLATED) as zf:
        for rel in [
            "weights",
            "reports",
            "artifacts",
            "rtl/scripts",
            "rtl/tb",
            "rtl/top",
            "rtl/layers",
            "rtl/common",
            "rtl/mem",
            "BASELINE_AUDIT.md",
            "DELIVERABLE_STATUS.md",
        ]:
            add_tree(zf, base / rel, base)

    print(output)


if __name__ == "__main__":
    main()
