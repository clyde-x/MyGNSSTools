#!/usr/bin/env python3
"""Create CSUPODS batch configs for the five DROL fusion candidates.

The script reads and writes XML as UTF-8 with a BOM.  CSUPODS accepts these
files, while a PowerShell text round-trip can corrupt the BOM and make the
application misleadingly report that MAINConfig.xml does not exist.
"""

from __future__ import annotations

import argparse
import re
import shutil
from datetime import date, timedelta
from pathlib import Path


BIN = Path(r"D:\csu\GNSS_MIX\CSUAPPS\CSUPODSApp\BIN")
TEMPLATE_BATCH = "DROL_FINAL_MIX_batch"
TEMPLATE_NAME = "DROL_FINAL_MIX"
METHODS = {
    "single_ant1": "DL0100LEO_{tag}.rnx",
    "single_ant2": "DL0200LEO_{tag}.rnx",
    "priority": "DROL_{tag}_priority.rnx",
    "screen0": "DROL_{tag}_screen0.rnx",
    "hybrid5": "DROL_{tag}_hybrid5.rnx",
}


def xml_text(path: Path) -> str:
    """Decode a CSUPODS XML file and reject malformed headers early."""
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    # Legacy CSUPODS templates often use GB2312 content despite their BOM.
    # Decode it faithfully, then write a self-consistent UTF-8 XML file.
    for encoding in ("utf-8", "gb18030"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise UnicodeDecodeError("xml", raw, 0, len(raw), f"Cannot decode {path}")
    # Some legacy templates have a space after the BOM, before ``<?xml``.
    # XML declarations must be the first node for strict parsers.
    text = text.lstrip("\ufeff \t\r\n")
    if not text.startswith("<?xml"):
        raise ValueError(f"Invalid XML declaration in {path}")
    return text


def write_xml(path: Path, text: str) -> None:
    """Always emit exactly one UTF-8 BOM followed by ``<?xml``."""
    if not text.startswith("<?xml"):
        raise ValueError(f"Refusing to write invalid XML declaration: {path}")
    # The source templates contain UTF-8 text but historically declare gb2312.
    # Keep declaration and bytes consistent so both strict XML parsers and
    # CSUPODS can consume the generated configuration.
    text, count = re.subn(
        r'(<\?xml\s+version=["\'][^"\']+["\']\s+encoding=["\'])[^"\']+',
        r'\g<1>utf-8',
        text,
        count=1,
        flags=re.I,
    )
    if count != 1:
        raise ValueError(f"XML encoding declaration missing: {path}")
    path.write_text(text, encoding="utf-8-sig", newline="")


def tag(text: str, name: str, value: str) -> str:
    text, n = re.subn(
        rf"(<{name}(?:\s+[^>]*)?>).*?(</{name}>)",
        rf"\g<1>{value}\g<2>", text, count=1, flags=re.S,
    )
    if n != 1:
        raise ValueError(f"<{name}> missing")
    return text


def days(start: date, end: date):
    while start <= end:
        yield start
        start += timedelta(days=1)


def doy_tag(day: date) -> str:
    return f"{day.year}{day.timetuple().tm_yday:03d}0"


def render_xml(text: str, *, batch: str, name: str, source_name: str, day: date, rinex: str) -> str:
    ymd = day.strftime("%Y%m%d")
    source_rel = f"{TEMPLATE_BATCH}/{source_name}"
    target_rel = f"{batch}/{name}"
    # Apply to every CSUPODS path family used by the daily template.
    for root in ("Config", "TempData", "LOG"):
        text = text.replace(
            f"RelativePathBIN/{root}/{source_rel}/",
            f"RelativePathBIN/{root}/{target_rel}/",
        )
    text = text.replace(
        f"RelativePathBIN/outputdata/{TEMPLATE_BATCH}/",
        f"RelativePathBIN/outputdata/{batch}/",
    )
    text = text.replace(source_name, name).replace(TEMPLATE_NAME, name.rsplit("_", 1)[0])
    text = re.sub(r"(?<=_)\d{8}(?=\.(?:sp3|rnx|dat|log))", ymd, text)
    text = re.sub(r"<ConfigFile>\d{8}</ConfigFile>", f"<ConfigFile>{ymd}</ConfigFile>", text)
    text = re.sub(r"(<ObservationDate>)\d{8}(</ObservationDate>)", rf"\g<1>{ymd}\g<2>", text)
    # Templates keep example InputRinexFile elements in comments.  Anchor to
    # a real element line so we never rewrite an example and leave the active
    # input path unchanged.
    text, count = re.subn(
        r"(?m)^(\s*<InputRinexFile>).*?(</InputRinexFile>)\s*$",
        rf"\g<1>RelativePathBIN/inputdata/Obs/DRO/{batch}/{rinex}\g<2>",
        text,
        count=1,
    )
    if count != 1:
        raise ValueError("Active InputRinexFile element missing")
    return text


def create(method: str, day: date, force: bool) -> None:
    batch = f"DROL_EVAL_{method}_batch"
    name = f"DROL_EVAL_{method}_{day:%Y%m%d}"
    source_name = f"{TEMPLATE_NAME}_{day:%Y%m%d}"
    source = BIN / "Config" / TEMPLATE_BATCH / source_name
    target = BIN / "Config" / batch / name
    if not source.is_dir():
        raise FileNotFoundError(source)
    if target.exists():
        if not force:
            raise FileExistsError(f"Use --force to replace {target}")
        shutil.rmtree(target)
    shutil.copytree(source, target)
    rinex = METHODS[method].format(tag=doy_tag(day))
    for xml in target.rglob("*.xml"):
        write_xml(xml, render_xml(xml_text(xml), batch=batch, name=name, source_name=source_name, day=day, rinex=rinex))
    main = target / "MAINConfig.xml"
    main_text = xml_text(main)
    main_text = tag(main_text, "ConfigFilePath", f"RelativePathBIN/Config/{batch}/{name}/")
    main_text = tag(main_text, "DynFitSwitch", "false")
    write_xml(main, main_text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-12-24")
    ap.add_argument("--end", default="2025-01-03")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--force", action="store_true")
    opt = ap.parse_args()
    start, end = date.fromisoformat(opt.start), date.fromisoformat(opt.end)
    for method in opt.methods.split(","):
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}")
        for day in days(start, end):
            create(method, day, opt.force)
            print(f"Generated: DROL_EVAL_{method}_batch/DROL_EVAL_{method}_{day:%Y%m%d}")


if __name__ == "__main__":
    main()
