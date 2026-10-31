"""Dated, self-contained result snapshots alongside the stable latest results."""
import datetime
import json
import os
import re
import shutil
import uuid
from pathlib import Path


def new_run_id() -> str:
    return f"{datetime.datetime.now():%H%M%S-%f}-{uuid.uuid4().hex[:6]}"


def timestamp() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _report_links(text: str, source_report: Path, out: Path,
                  archived_report: Path, archive: Path) -> str:
    text = re.sub(r"\nArchived copy: \[[^\n]*\n?", "\n", text)

    def replace(match: re.Match[str]) -> str:
        target = match.group(2)
        if target.startswith(("https://", "http://", "mailto:", "#")):
            return match.group(0)
        source = (source_report / target).resolve()
        if source.is_relative_to(source_report.resolve()):
            destination = archived_report / source.relative_to(source_report.resolve())
        elif source.is_relative_to((out / "artifacts").resolve()):
            destination = source
        elif source.is_relative_to(out.resolve()):
            destination = archive / source.relative_to(out.resolve())
        else:
            return match.group(0)
        relative = Path(os.path.relpath(destination, archived_report)).as_posix()
        return f"{match.group(1)}{relative}{match.group(3)}"

    return re.sub(r"(!?\[[^\]]*\]\()([^)\n]+)(\))", replace, text)


def _index(out: Path) -> None:
    rows = []
    for path in sorted((out / "runs").glob("*/*/snapshot.json"), reverse=True):
        meta = json.loads(path.read_text(encoding="utf-8"))
        report = path.parent / "report" / "report.md"
        target = report if report.exists() else path
        relative = Path(os.path.relpath(target, out / "report")).as_posix()
        rows.append(f"| {meta['created_at']} | {meta['reason']} | "
                    f"{', '.join(meta['datasets'])} | [{meta['snapshot_id']}]({relative}) |")
    lines = ["---", "title: Run history",
             "description: Dated snapshots of Quibble results and reports.", "---", "",
             "## Run history", "",
             "The latest report is [report.md](report.md). Snapshots preserve earlier results; "
             "snapshots taken before training can include the previous report, while completed "
             "training snapshots contain raw results only.", "",
             "| Created | Reason | Datasets | Snapshot |",
             "|---|---|---|---|", *rows, ""]
    legacy = out / "report" / "legacy"
    if legacy.exists():
        lines += ["## Legacy reports", ""]
        for path in sorted(legacy.glob("*/report/report.md")):
            relative = path.relative_to(out / "report").as_posix()
            lines.append(f"- [{path.parent.parent.name}]({relative})")
        lines.append("")
    directory = out / "report"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "history.md").write_text("\n".join(lines), encoding="utf-8")


def snapshot(out: str | Path, reason: str, report_dir: str | Path | None = None,
             include_report: bool = True) -> Path | None:
    """Preserve generated datasets, analysis, report and figures; never copy history or caches."""
    out = Path(out)
    report_dir = Path(report_dir) if report_dir is not None else out / "report"
    datasets = sorted(p for p in out.iterdir()
                      if p.is_dir() and p.name not in ("runs", "report")
                      and list(p.glob("*_ss*_ms*.json"))) if out.exists() else []
    report_files = []
    if include_report:
        report_files = ([report_dir / "report.md"] if (report_dir / "report.md").exists() else [])
        report_files += sorted(report_dir.glob("fig_*.png"))
        report_files += [report_dir / name for name in (
            "participation.csv", "backend_provenance.json", "training_protocol.csv",
            "stage_costs.csv", "tier_comparison.csv", "tier_comparison.json")
                         if (report_dir / name).exists()]
    if not datasets and not report_files:
        return None
    snapshot_id = new_run_id()
    archive = out / "runs" / f"{datetime.date.today():%Y-%m-%d}" / snapshot_id
    archive.mkdir(parents=True)
    run_ids = set()
    invocation = out / "invocation.json"
    if invocation.exists():
        shutil.copy2(invocation, archive / invocation.name)
    feature_stage = out / "relational-feature-stage.json"
    if feature_stage.exists():
        shutil.copy2(feature_stage, archive / feature_stage.name)
    for directory in datasets:
        shutil.copytree(directory, archive / directory.name)
        for path in directory.glob("*_ss*_ms*.json"):
            record = json.loads(path.read_text())
            if record.get("run_id"):
                run_ids.add(record["run_id"])
    archived_report = archive / "report"
    if report_files:
        archived_report.mkdir()
    for path in report_files:
        if path.name == "report.md":
            text = _report_links(path.read_text(encoding="utf-8"), report_dir, out, archived_report, archive)
            (archived_report / path.name).write_text(text, encoding="utf-8")
        else:
            shutil.copy2(path, archived_report / path.name)
    meta = {"snapshot_id": snapshot_id, "created_at": timestamp(), "reason": reason,
            "datasets": [p.name for p in datasets], "run_ids": sorted(run_ids),
            "source": str(out.resolve())}
    (archive / "snapshot.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _index(out)
    return archive
