"""Extract a hierarchical tag list from an XML file.

This creates a Markdown outline of tags in document order, preserving
parent/child structure and listing attribute names seen per tag.
Optionally include a sample of element text and attribute values.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class TagNode:
	name: str
	count: int = 0
	attrs: set[str] = field(default_factory=set)
	attr_samples: dict[str, str] = field(default_factory=dict)
	sample_text: str | None = None
	children: "OrderedDict[str, TagNode]" = field(default_factory=OrderedDict)

	def get_child(self, name: str) -> "TagNode":
		if name not in self.children:
			self.children[name] = TagNode(name=name)
		return self.children[name]


def _truncate(value: str, max_len: int) -> str:
	if len(value) <= max_len:
		return value
	return value[: max_len - 1] + "…"


def build_tag_tree(
	xml_path: Path,
	include_values: bool,
	max_text: int,
	max_attr: int,
) -> TagNode:
	root = TagNode(name="ROOT")
	stack: list[TagNode] = [root]

	for event, elem in ET.iterparse(xml_path, events=("start", "end")):
		if event == "start":
			parent = stack[-1]
			node = parent.get_child(elem.tag)
			node.count += 1
			if elem.attrib:
				node.attrs.update(elem.attrib.keys())
				if include_values:
					for key, value in elem.attrib.items():
						if key not in node.attr_samples:
							node.attr_samples[key] = _truncate(str(value), max_attr)
			stack.append(node)
		elif event == "end":
			if include_values and elem.text and not stack[-1].sample_text:
				text = elem.text.strip()
				if text:
					stack[-1].sample_text = _truncate(text, max_text)
			stack.pop()
			elem.clear()

	return root


def _format_attrs(attrs: Iterable[str], attr_samples: dict[str, str], include_values: bool) -> str:
	items = sorted(attrs)
	if not items:
		return ""
	if not include_values:
		return f" attrs: [{', '.join(items)}]"

	rendered: list[str] = []
	for key in items:
		if key in attr_samples:
			rendered.append(f"{key}={attr_samples[key]}")
		else:
			rendered.append(key)
	return f" attrs: [{', '.join(rendered)}]"


def write_markdown(
	root: TagNode,
	output_path: Path,
	source_path: Path,
	include_values: bool,
) -> None:
	lines: list[str] = []
	lines.append(f"# XML Tag Outline\n")
	lines.append(f"Source: {source_path}\n")
	lines.append("")

	def walk(node: TagNode, depth: int) -> None:
		if node.name != "ROOT":
			indent = "  " * depth
			line = (
				f"{indent}- {node.name} (count: {node.count})"
				f"{_format_attrs(node.attrs, node.attr_samples, include_values)}"
			)
			if include_values and node.sample_text:
				line += f" value: {node.sample_text}"
			lines.append(line)
		for child in node.children.values():
			walk(child, depth + 1)

	walk(root, 0)
	output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
	parser = argparse.ArgumentParser(description="Create a tag outline for XML.")
	parser.add_argument("input", help="Path to XML file")
	parser.add_argument(
		"-o",
		"--output",
		help="Path to output Markdown file (default: same name with .tags.md)",
	)
	parser.add_argument(
		"--values",
		action="store_true",
		default=True,
		help="Include a sample of element text and attribute values (default: on)",
	)
	parser.add_argument(
		"--no-values",
		action="store_false",
		dest="values",
		help="Disable value sampling in the output",
	)
	parser.add_argument(
		"--max-text",
		type=int,
		default=80,
		help="Max length for text samples when --values is used (default: 80)",
	)
	parser.add_argument(
		"--max-attr",
		type=int,
		default=60,
		help="Max length for attribute samples when --values is used (default: 60)",
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
		else input_path.with_suffix(".tags.md")
	)

	try:
		root = build_tag_tree(
			input_path,
			include_values=args.values,
			max_text=args.max_text,
			max_attr=args.max_attr,
		)
		write_markdown(root, output_path, input_path, include_values=args.values)
	except Exception as exc:  # noqa: BLE001
		print(f"Failed to parse {input_path}: {exc}", file=sys.stderr)
		return 1

	print(f"Wrote tag outline to: {output_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
