"""Reads back the .xlsx files the app writes, so the tests can check their content.

Only understands what rollcall.xlsx writes (inline strings and numbers).
"""
import re
import zipfile
import xml.etree.ElementTree as ET

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}


def column_index(ref: str) -> int:
    """"A1" -> 0, "AA7" -> 26."""
    index = 0
    for letter in re.match(r"[A-Z]+", ref).group():
        index = index * 26 + ord(letter) - ord("A") + 1
    return index - 1


def read(path):
    """{sheet name: list of rows}, each row a list of (value, style index)."""
    with zipfile.ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        names = [s.get("name") for s in workbook.find("m:sheets", NS)]
        sheets = {}
        for number, name in enumerate(names, start=1):
            root = ET.fromstring(archive.read(f"xl/worksheets/sheet{number}.xml"))
            rows = []
            for row in root.find("m:sheetData", NS):
                assert int(row.get("r")) == len(rows) + 1  # rows are consecutive
                cells = []
                for cell in row:
                    col = column_index(cell.get("r"))
                    cells += [(None, None)] * (col - len(cells))  # empty cells were left out
                    if cell.get("t") == "inlineStr":
                        value = cell.find("m:is/m:t", NS).text or ""
                    else:
                        value = float(cell.find("m:v", NS).text)
                    cells.append((value, int(cell.get("s"))))
                rows.append(cells)
            sheets[name] = rows
        return sheets


def values(path, sheet):
    """Only the values of one sheet, without styles."""
    return [[value for value, _ in row] for row in read(path)[sheet]]
