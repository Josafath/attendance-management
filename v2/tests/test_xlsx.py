import zipfile
import xml.etree.ElementTree as ET
from datetime import date, datetime

import pytest

from rollcall import xlsx
from rollcall.xlsx import Percent, Workbook, column_letter, excel_serial
import xlsx_reader

STYLE = {kind: i * 2 for i, kind in enumerate(xlsx.KINDS)}  # non-bold style index per kind


@pytest.mark.parametrize("index, letters", [(0, "A"), (25, "Z"), (26, "AA"), (51, "AZ"),
                                            (701, "ZZ"), (702, "AAA")])
def test_column_letters(index, letters):
    assert column_letter(index) == letters
    assert xlsx_reader.column_index(letters + "1") == index


def test_excel_serial_numbers():
    assert excel_serial(date(1900, 3, 1)) == 61           # a value every spreadsheet agrees on
    assert excel_serial(datetime(2026, 1, 12, 12, 0)) == pytest.approx(46034.5)


def test_every_xml_part_is_well_formed(tmp_path):
    book = Workbook()
    book.add_sheet("One").append(["a", 1])
    book.add_sheet("Two").append([2.5])
    path = book.save(tmp_path / "book.xlsx")
    with zipfile.ZipFile(path) as archive:
        assert set(archive.namelist()) == {
            "[Content_Types].xml", "_rels/.rels", "xl/workbook.xml", "xl/_rels/workbook.xml.rels",
            "xl/styles.xml", "xl/worksheets/sheet1.xml", "xl/worksheets/sheet2.xml"}
        for name in archive.namelist():
            ET.fromstring(archive.read(name))  # raises if any XML is broken


def test_values_and_their_formats(tmp_path):
    book = Workbook()
    sheet = book.add_sheet("Data")
    sheet.append(["text", 42, 2.5, Percent(0.755), datetime(2026, 1, 12, 8, 30), date(2026, 1, 12),
                  None, True, float("nan")])
    sheet.append(["Header"], bold=True)
    cells = xlsx_reader.read(book.save(tmp_path / "f.xlsx"))["Data"]

    row = cells[0]
    assert row[0] == ("text", STYLE["general"])
    assert row[1] == (42.0, STYLE["general"])
    assert row[2] == (2.5, STYLE["general"])
    assert row[3] == (pytest.approx(0.755), STYLE["percent"])
    assert row[4] == (pytest.approx(46034.354166, abs=1e-5), STYLE["datetime"])
    assert row[5] == (46034.0, STYLE["date"])
    assert row[6] == (None, None)            # None -> empty cell
    assert row[7] == ("Yes", STYLE["general"])  # not the number 1
    assert len(row) == 8                     # NaN -> empty cell (left out at the end)
    assert cells[1][0] == ("Header", STYLE["general"] + 1)  # bold


def test_text_is_escaped_and_never_a_formula(tmp_path):
    book = Workbook()
    tricky = ['=HYPERLINK("http://evil")', "<b>Ana & Beto</b>", "  spaces  ", "tab\there", "bad\x00\x07chars",
              "Peña Nuñez"]
    book.add_sheet("Text").append(tricky)
    path = book.save(tmp_path / "t.xlsx")
    read = xlsx_reader.values(path, "Text")[0]
    assert read[:4] == tricky[:4]            # stored exactly, as text
    assert read[4] == "badchars"             # control characters removed (illegal in XML)
    assert read[5] == "Peña Nuñez"           # accents survive (UTF-8)
    with zipfile.ZipFile(path) as archive:
        assert b"<f>" not in archive.read("xl/worksheets/sheet1.xml")  # no formulas at all


def test_widths_and_frozen_header(tmp_path):
    book = Workbook()
    sheet = book.add_sheet("S")
    sheet.append(["h"])
    sheet.set_widths(10, 20)
    sheet.freeze(1)
    with zipfile.ZipFile(book.save(tmp_path / "w.xlsx")) as archive:
        xml = archive.read("xl/worksheets/sheet1.xml").decode()
    assert 'ySplit="1"' in xml and 'state="frozen"' in xml
    assert '<col min="2" max="2" width="20" customWidth="1"/>' in xml


@pytest.mark.parametrize("name", ["", "x" * 32, "a/b", "what?", "[x]"])
def test_invalid_sheet_names(name):
    with pytest.raises(ValueError):
        Workbook().add_sheet(name)


def test_sheet_names_are_unique_ignoring_case():
    book = Workbook()
    book.add_sheet("Students")
    with pytest.raises(ValueError):
        book.add_sheet("STUDENTS")


def test_sheet_name_with_quotes_and_ampersand(tmp_path):
    book = Workbook()
    book.add_sheet('Ana & "Beto"').append(["x"])
    assert list(xlsx_reader.read(book.save(tmp_path / "q.xlsx"))) == ['Ana & "Beto"']


def test_unsupported_values_and_huge_text_fail_clearly(tmp_path):
    book = Workbook()
    book.add_sheet("S").append([object()])
    with pytest.raises(TypeError):
        book.save(tmp_path / "a.xlsx")
    book = Workbook()
    book.add_sheet("S").append(["x" * (xlsx.MAX_TEXT + 1)])
    with pytest.raises(ValueError):
        book.save(tmp_path / "b.xlsx")


def test_failed_save_leaves_no_file(tmp_path):
    book = Workbook()
    book.add_sheet("S").append([object()])
    with pytest.raises(TypeError):
        book.save(tmp_path / "broken.xlsx")
    assert list(tmp_path.iterdir()) == []  # neither the file nor the temporary one


def test_empty_workbook_is_refused(tmp_path):
    with pytest.raises(ValueError):
        Workbook().save(tmp_path / "empty.xlsx")
