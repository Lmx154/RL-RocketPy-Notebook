"""Convert OpenRocket .ork files to XML.

OpenRocket .ork files are either:
- Plain XML (already the OpenRocket document), or
- A ZIP archive containing the XML document.

This script extracts the XML in a lossless way and writes it to a .xml file.
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

XML_PREFIXES = (b"<?xml", b"<openrocket")


def _is_xml_bytes(data: bytes) -> bool:
	head = data.lstrip()
	return any(head.startswith(prefix) for prefix in XML_PREFIXES)


def _read_xml_from_zip(zip_path: Path) -> bytes:
	with zipfile.ZipFile(zip_path, "r") as zf:
		# Prefer explicit XML/ORK names, then any file that looks like XML
		candidates = [
			name
			for name in zf.namelist()
			if name.lower().endswith((".ork", ".xml"))
		]
		for name in candidates:
			data = zf.read(name)
			if _is_xml_bytes(data):
				return data

		# Fallback: scan all files
		for name in zf.namelist():
			data = zf.read(name)
			if _is_xml_bytes(data):
				return data

	raise ValueError("No XML content found inside ORK archive.")


def ork_to_xml_bytes(ork_path: Path) -> bytes:
	raw = ork_path.read_bytes()

	if _is_xml_bytes(raw):
		return raw

	if zipfile.is_zipfile(ork_path):
		return _read_xml_from_zip(ork_path)

	raise ValueError("Unrecognized ORK format: not XML and not a ZIP archive.")


def main() -> int:
	parser = argparse.ArgumentParser(description="Convert .ork file to XML.")
	parser.add_argument("input", help="Path to .ork file")
	parser.add_argument(
		"-o",
		"--output",
		help="Path to output .xml file (default: same name with .xml)",
	)

	args = parser.parse_args()
	input_arg = Path(args.input).expanduser()
	input_path = input_arg.resolve()

	if not input_path.exists() and not input_arg.is_absolute():
		repo_root = Path(__file__).resolve().parent.parent
		alt_path = (repo_root / input_arg).resolve()
		if alt_path.exists():
			input_path = alt_path

	if not input_path.exists():
		print(f"Input file not found: {input_path}", file=sys.stderr)
		return 2

	output_path = (
		Path(args.output).expanduser().resolve()
		if args.output
		else input_path.with_suffix(".xml")
	)

	try:
		xml_bytes = ork_to_xml_bytes(input_path)
	except Exception as exc:  # noqa: BLE001
		print(f"Failed to convert {input_path}: {exc}", file=sys.stderr)
		return 1

	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_bytes(xml_bytes)

	print(f"Wrote XML to: {output_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

