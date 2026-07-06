import json
import tempfile
import unittest
from pathlib import Path

from quality import validate_corpus_chunks, write_quality_report


_TS = "2026-06-27T00:00:00"

_CLEAN = {
    "doc_id": "clean.txt",
    "chunk_index": 0,
    "text": (
        "Substance is that which is in itself and is conceived through itself. "
        "The order and connection of ideas is the same as the order and "
        "connection of things. This is ordinary prose with stable spacing."
    ),
}

_OCR_GARBAGE = {
    "doc_id": "scan.txt",
    "chunk_index": 1,
    "text": (
        "QWRTYPLK 9F3KZ ### @@@ !!!!! ????? X7B9Q X7B9Q X7B9Q "
        "X7B9Q X7B9Q X7B9Q zzzzzzz bcdfrg hjklmnp qqqqqqq"
    ),
}

_STAMP = {
    "doc_id": "stamp.txt",
    "chunk_index": 2,
    "text": "PROPERTY OF UNIVERSITY LIBRARY\nDEPARTMENT OF PHILOSOPHY",
}

_PUBLISHER_CATALOGUE = {
    "doc_id": "catalogue.txt",
    "chunk_index": 3,
    "text": (
        "Publisher's Catalogue\n"
        "Recent Publications and Selected Titles\n"
        "The History of Thought. By A. Scholar. Cloth, $2.00 net.\n"
        "Studies in Modern Life. By B. Writer. Volume II.\n"
        "A Complete List of Books and Forthcoming Editions.\n"
    ),
}


class TestCorpusQualityRules(unittest.TestCase):

    def test_clean_document_passes_without_warning(self):
        accepted, report = validate_corpus_chunks([_CLEAN], validation_timestamp=_TS)

        self.assertEqual(accepted, [_CLEAN])
        self.assertEqual(report.passed_chunks, 1)
        self.assertEqual(report.warning_chunks, 0)
        self.assertEqual(report.rejected_chunks, 0)

    def test_ocr_garbage_is_rejected(self):
        accepted, report = validate_corpus_chunks([_OCR_GARBAGE], validation_timestamp=_TS)

        self.assertEqual(accepted, [])
        self.assertEqual(report.rejected_chunks, 1)
        self.assertTrue(
            {"excessive_symbol_density", "suspicious_ocr_noise"}
            & set(report.rejection_reasons)
        )

    def test_uppercase_stamp_fragment_is_rejected(self):
        accepted, report = validate_corpus_chunks([_STAMP], validation_timestamp=_TS)

        self.assertEqual(accepted, [])
        self.assertEqual(report.rejection_reasons["uppercase_stamp_fragment"], 1)

    def test_publisher_catalogue_is_rejected(self):
        accepted, report = validate_corpus_chunks(
            [_PUBLISHER_CATALOGUE],
            validation_timestamp=_TS,
        )

        self.assertEqual(accepted, [])
        self.assertEqual(report.rejection_reasons["publisher_back_matter"], 1)

    def test_mixed_quality_corpus_keeps_clean_and_reports_rejections(self):
        accepted, report = validate_corpus_chunks(
            [_CLEAN, _OCR_GARBAGE, _PUBLISHER_CATALOGUE],
            validation_timestamp=_TS,
        )

        self.assertEqual(accepted, [_CLEAN])
        self.assertEqual(report.total_chunks, 3)
        self.assertEqual(report.accepted_chunks, 1)
        self.assertEqual(report.rejected_chunks, 2)

    def test_quality_report_is_machine_readable_json(self):
        _, report = validate_corpus_chunks([_CLEAN, _OCR_GARBAGE], validation_timestamp=_TS)

        with tempfile.TemporaryDirectory() as tmp:
            path = write_quality_report(report, Path(tmp))
            data = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(data["quality_report_version"], "1.0")
        self.assertEqual(data["total_chunks"], 2)
        self.assertIn("rejection_reasons", data)
        self.assertIn("chunks", data)


if __name__ == "__main__":
    unittest.main()
