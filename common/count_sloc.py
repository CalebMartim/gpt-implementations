import json
import os
from pathlib import Path

HARNESS_MARKER = "// === BENCHMARK HARNESS ==="
CODE_EXTENSIONS = {".py", ".cu", ".c", ".cpp", ".h", ".hpp"}


def count_text_sloc(text: str) -> int:
    """Count physical source lines of code excluding blanks, comments, and docstrings."""
    sloc = 0
    in_block_comment = False
    in_triple_quote: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # Handle ongoing Python triple-quoted docstring
        if in_triple_quote is not None:
            end_idx = line.find(in_triple_quote)
            if end_idx != -1:
                rest = line[end_idx + 3 :].strip()
                in_triple_quote = None
                if not rest or rest.startswith("#") or rest.startswith("//"):
                    continue
                line = rest
            else:
                continue

        # Handle ongoing C/CUDA block comment /* ... */
        if in_block_comment:
            end_idx = line.find("*/")
            if end_idx != -1:
                line = line[end_idx + 2 :].strip()
                in_block_comment = False
                if not line:
                    continue
            else:
                continue

        # Remove inline /* ... */ segments on the current line
        while "/*" in line:
            start_idx = line.find("/*")
            end_idx = line.find("*/", start_idx + 2)
            if end_idx != -1:
                line = (line[:start_idx] + " " + line[end_idx + 2 :]).strip()
            else:
                line = line[:start_idx].strip()
                in_block_comment = True
                break

        if not line:
            continue

        # Check for single-line comments
        if line.startswith("#") or line.startswith("//"):
            continue

        # Check for Python triple-quoted docstring starting the line
        if line.startswith('"""') or line.startswith("'''"):
            quote = line[:3]
            rest = line[3:]
            end_idx = rest.find(quote)
            if end_idx != -1:
                after = rest[end_idx + 3 :].strip()
                if not after or after.startswith("#"):
                    continue
                line = after
            else:
                in_triple_quote = quote
                continue

        sloc += 1

    return sloc


def count_file_sloc(filepath: str) -> int:
    """Count physical SLOC in a single source file."""
    content = Path(filepath).read_text(encoding="utf-8")
    return count_text_sloc(content)


def count_implementation_sloc(impl_dir: str) -> dict:
    """Count core, harness, and total SLOC for a single implementation directory."""
    impl_path = Path(impl_dir)
    sloc_core = 0
    sloc_harness = 0
    files_report: dict[str, int] = {}

    if not impl_path.exists():
        return {
            "sloc_core": 0,
            "sloc_harness": 0,
            "sloc_total": 0,
            "files": files_report,
        }

    for entry in sorted(impl_path.iterdir()):
        if not entry.is_file():
            continue
        if entry.name == "__init__.py" or entry.suffix not in CODE_EXTENSIONS:
            continue

        content = entry.read_text(encoding="utf-8")
        if entry.name == "gpt_cuda.cu" and HARNESS_MARKER in content:
            core_part, harness_part = content.split(HARNESS_MARKER, maxsplit=1)
            c_sloc = count_text_sloc(core_part)
            h_sloc = count_text_sloc(harness_part)
            sloc_core += c_sloc
            sloc_harness += h_sloc
            files_report[f"{entry.name} (core)"] = c_sloc
            files_report[f"{entry.name} (harness)"] = h_sloc
        elif "benchmark" in entry.name:
            f_sloc = count_text_sloc(content)
            sloc_harness += f_sloc
            files_report[entry.name] = f_sloc
        else:
            f_sloc = count_text_sloc(content)
            sloc_core += f_sloc
            files_report[entry.name] = f_sloc

    return {
        "sloc_core": sloc_core,
        "sloc_harness": sloc_harness,
        "sloc_total": sloc_core + sloc_harness,
        "files": files_report,
    }


def count_all_implementations(root_dir: str = "implementations") -> dict:
    """Return SLOC report for every implementation subdirectory in root_dir."""
    root_path = Path(root_dir)
    if not root_path.exists():
        repo_root = Path(__file__).resolve().parent.parent
        candidate = repo_root / root_dir
        if candidate.exists():
            root_path = candidate

    report: dict[str, dict] = {}
    if not root_path.exists():
        return report

    for subdir in sorted(root_path.iterdir()):
        if subdir.is_dir() and not subdir.name.startswith((".", "_")):
            report[subdir.name] = count_implementation_sloc(str(subdir))
    return report


if __name__ == "__main__":
    report = count_all_implementations()
    print(json.dumps(report, indent=2))
