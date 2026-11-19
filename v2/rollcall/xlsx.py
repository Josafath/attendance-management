"""A small Excel (.xlsx) writer that only uses Python's standard library.

An .xlsx file is a ZIP archive of XML files (the "Office Open XML" format):

    [Content_Types].xml          what kind of file each part is
    _rels/.rels                  where the workbook is
    xl/workbook.xml              the list of sheets
    xl/_rels/workbook.xml.rels   where each sheet and the styles are
    xl/styles.xml                fonts and number formats (bold, dates, %)
    xl/worksheets/sheet1.xml     the cells of each sheet

This module writes only what the app needs: text, numbers, dates, percentages,
bold rows, column widths and a frozen header. It knows nothing about students:
export.py decides WHAT goes in the file, this module decides HOW it's written.

Text is always stored as text, never as a formula, so a name like "=1+1" can't
run anything when the file is opened (a known risk of CSV exports).
"""
from __future__ import annotations

import math
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Union
from xml.sax.saxutils import escape


class Percent(float):
    """A fraction shown as a percentage: Percent(0.755) appears as 75.5 %."""


Value = Union[str, int, float, Percent, datetime, date, None]

# Excel stores dates as days since 1899-12-30 (a historical quirk inherited from Lotus 1-2-3).
EXCEL_EPOCH = datetime(1899, 12, 30)

# Number formats: ids 164+ are custom ones, defined in styles.xml.
FORMATS = {"general": 0, "datetime": 164, "percent": 165, "date": 166}
FORMAT_CODES = {164: "yyyy-mm-dd hh:mm", 165: "0.0%", 166: "yyyy-mm-dd"}
KINDS = list(FORMATS)  # style index = kind position * 2 + (1 if bold)

# Characters that XML 1.0 doesn't allow, even escaped (control characters).
_ILLEGAL_XML = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_ILLEGAL_SHEET_NAME = re.compile(r"[\[\]:*?/\\]")
MAX_TEXT = 32_767  # Excel's limit for one cell


@dataclass
class Sheet:
    name: str
    rows: list[tuple[list[Value], bool]] = field(default_factory=list)  # (values, bold)
    widths: list[float] = field(default_factory=list)
    frozen_rows: int = 0

    def append(self, values: Iterable[Value] = (), bold: bool = False) -> None:
        """Adds a row. An empty call adds a blank row."""
        self.rows.append((list(values), bold))

    def set_widths(self, *widths: float) -> None:
        """Column widths in characters, from column A onwards."""
        self.widths = list(widths)

    def freeze(self, rows: int) -> None:
        """Keeps the first `rows` rows visible while scrolling (e.g. the header)."""
        self.frozen_rows = rows


class Workbook:
    def __init__(self) -> None:
        self.sheets: list[Sheet] = []

    def add_sheet(self, name: str) -> Sheet:
        _check_sheet_name(name, [s.name for s in self.sheets])
        sheet = Sheet(name)
        self.sheets.append(sheet)
        return sheet

    def save(self, path: Union[str, Path]) -> Path:
        """Writes the file. It's written to a temporary file first and then
        renamed, so a failure never leaves a broken .xlsx behind."""
        if not self.sheets:
            raise ValueError("A workbook needs at least one sheet.")
        path = Path(path)
        handle, temp = tempfile.mkstemp(suffix=".xlsx.part", dir=path.parent)
        os.close(handle)
        try:
            with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as archive:
                for name, content in self._parts().items():
                    archive.writestr(name, content)
            os.replace(temp, path)
        except BaseException:
            os.unlink(temp)
            raise
        return path

    # ---------- the XML parts ----------

    def _parts(self) -> dict[str, str]:
        count = len(self.sheets)
        parts = {
            "[Content_Types].xml": _content_types(count),
            "_rels/.rels": _ROOT_RELS,
            "xl/workbook.xml": _workbook(self.sheets),
            "xl/_rels/workbook.xml.rels": _workbook_rels(count),
            "xl/styles.xml": _STYLES,
        }
        for number, sheet in enumerate(self.sheets, start=1):
            parts[f"xl/worksheets/sheet{number}.xml"] = _sheet_xml(sheet)
        return parts


# ---------------------------------------------------------------- cells

def column_letter(index: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA, 701 -> ZZ, 702 -> AAA (like a base-26 number without zero)."""
    letters = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return letters


def excel_serial(value: Union[datetime, date]) -> float:
    """A date as Excel's day number: 2026-01-12 08:00 -> 46034.333..."""
    if not isinstance(value, datetime):
        value = datetime(value.year, value.month, value.day)
    return (value.replace(tzinfo=None) - EXCEL_EPOCH).total_seconds() / 86400


def _cell(ref: str, value: Value, bold: bool) -> str:
    """One <c> element. Returns "" for empty cells (they're simply left out)."""
    def style(kind: str) -> int:
        return KINDS.index(kind) * 2 + (1 if bold else 0)

    if value is None:
        return ""
    if isinstance(value, bool):
        # bool is a subclass of int: without this, True would be written as 1.
        value = "Yes" if value else "No"
    if isinstance(value, str):
        if len(value) > MAX_TEXT:
            raise ValueError(f"Cell {ref} has {len(value)} characters; Excel allows {MAX_TEXT}.")
        text = escape(_ILLEGAL_XML.sub("", value))
        # inlineStr keeps the text inside the cell; xml:space keeps leading/trailing spaces.
        return f'<c r="{ref}" s="{style("general")}" t="inlineStr"><is><t xml:space="preserve">{text}</t></is></c>'
    if isinstance(value, datetime):
        return f'<c r="{ref}" s="{style("datetime")}"><v>{excel_serial(value)!r}</v></c>'
    if isinstance(value, date):
        return f'<c r="{ref}" s="{style("date")}"><v>{excel_serial(value)!r}</v></c>'
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return ""  # Excel has no NaN or infinity
        kind = "percent" if isinstance(value, Percent) else "general"
        number = repr(float(value)) if isinstance(value, float) else str(value)
        return f'<c r="{ref}" s="{style(kind)}"><v>{number}</v></c>'
    raise TypeError(f"Cell {ref}: can't write a {type(value).__name__}.")


def _sheet_xml(sheet: Sheet) -> str:
    parts = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
             '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">']
    if sheet.frozen_rows:
        top_left = f"A{sheet.frozen_rows + 1}"
        parts.append(f'<sheetViews><sheetView workbookViewId="0"><pane ySplit="{sheet.frozen_rows}" '
                     f'topLeftCell="{top_left}" activePane="bottomLeft" state="frozen"/>'
                     f'</sheetView></sheetViews>')
    if sheet.widths:
        cols = "".join(f'<col min="{i}" max="{i}" width="{w}" customWidth="1"/>'
                       for i, w in enumerate(sheet.widths, start=1))
        parts.append(f"<cols>{cols}</cols>")
    parts.append("<sheetData>")
    for row_number, (values, bold) in enumerate(sheet.rows, start=1):
        cells = "".join(_cell(f"{column_letter(col)}{row_number}", value, bold)
                        for col, value in enumerate(values))
        parts.append(f'<row r="{row_number}">{cells}</row>')
    parts.append("</sheetData></worksheet>")
    return "".join(parts)


def _attribute(text: str) -> str:
    """Text escaped for an XML attribute: & < > and also the double quote."""
    return escape(text, {'"': "&quot;"})


def _check_sheet_name(name: str, existing: list[str]) -> None:
    # Excel's rules for sheet names.
    if not 1 <= len(name) <= 31:
        raise ValueError("A sheet name must have between 1 and 31 characters.")
    if _ILLEGAL_SHEET_NAME.search(name):
        raise ValueError(r"A sheet name can't contain [ ] : * ? / \ ")
    if name.lower() in (e.lower() for e in existing):
        raise ValueError(f"There is already a sheet called {name!r}.")


# ---------------------------------------------------------------- fixed parts

def _content_types(sheet_count: int) -> str:
    sheets = "".join(
        f'<Override PartName="/xl/worksheets/sheet{n}.xml" '
        f'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for n in range(1, sheet_count + 1))
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/styles.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
            f"{sheets}</Types>")


_ROOT_RELS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" '
    'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
    'Target="xl/workbook.xml"/></Relationships>'
)


def _workbook(sheets: list[Sheet]) -> str:
    # The name goes inside an attribute, so double quotes must be escaped too.
    entries = "".join(f'<sheet name="{_attribute(s.name)}" sheetId="{n}" r:id="rId{n}"/>'
                      for n, s in enumerate(sheets, start=1))
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f"<sheets>{entries}</sheets></workbook>")


def _workbook_rels(sheet_count: int) -> str:
    sheets = "".join(
        f'<Relationship Id="rId{n}" '
        f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        f'Target="worksheets/sheet{n}.xml"/>' for n in range(1, sheet_count + 1))
    styles = (f'<Relationship Id="rId{sheet_count + 1}" '
              'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
              'Target="styles.xml"/>')
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f"{sheets}{styles}</Relationships>")


def _styles() -> str:
    num_fmts = "".join(f'<numFmt numFmtId="{i}" formatCode="{code}"/>' for i, code in FORMAT_CODES.items())

    def xf(kind: str, bold: int) -> str:
        # applyNumberFormat / applyFont tell Excel to use this style's format and font.
        extra = ' applyNumberFormat="1"' if FORMATS[kind] else ""
        extra += ' applyFont="1"' if bold else ""
        return f'<xf numFmtId="{FORMATS[kind]}" fontId="{bold}" fillId="0" borderId="0"{extra}/>'

    # One cell style per (number format, bold) pair, in the order _cell() expects.
    xfs = "".join(xf(kind, bold) for kind in KINDS for bold in (0, 1))
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f'<numFmts count="{len(FORMAT_CODES)}">{num_fmts}</numFmts>'
            '<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font>'
            '<font><b/><sz val="11"/><name val="Calibri"/></font></fonts>'
            '<fills count="2"><fill><patternFill patternType="none"/></fill>'
            '<fill><patternFill patternType="gray125"/></fill></fills>'
            '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
            '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
            f'<cellXfs count="{len(KINDS) * 2}">{xfs}</cellXfs>'
            '</styleSheet>')


_STYLES = _styles()
