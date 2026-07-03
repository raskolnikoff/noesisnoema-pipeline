"""Tests for noema_audit.jcs (RFC 8785 canonicalizer, numbers-as-strings rule)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from noema_audit.jcs import CanonicalizationError, canonicalize


class TestCanonicalizeScalars(unittest.TestCase):

    def test_none(self):
        self.assertEqual(canonicalize(None), b"null")

    def test_true(self):
        self.assertEqual(canonicalize(True), b"true")

    def test_false(self):
        self.assertEqual(canonicalize(False), b"false")

    def test_integer(self):
        self.assertEqual(canonicalize(42), b"42")

    def test_negative_integer(self):
        self.assertEqual(canonicalize(-7), b"-7")

    def test_zero(self):
        self.assertEqual(canonicalize(0), b"0")

    def test_string_minimal_escaping(self):
        self.assertEqual(canonicalize("hello"), b'"hello"')

    def test_string_quote_and_backslash_escaped(self):
        self.assertEqual(canonicalize('a"b\\c'), b'"a\\"b\\\\c"')

    def test_string_control_char_escaped(self):
        self.assertEqual(canonicalize("a\nb"), b'"a\\nb"')

    def test_string_unicode_left_raw(self):
        # RFC 8785 does not escape non-ASCII; only JSON-mandatory escapes apply.
        self.assertEqual(canonicalize("タカ"), '"タカ"'.encode("utf-8"))


class TestCanonicalizeFloatRejection(unittest.TestCase):

    def test_float_raises(self):
        with self.assertRaises(CanonicalizationError):
            canonicalize(0.87)

    def test_float_nested_in_dict_raises(self):
        with self.assertRaises(CanonicalizationError):
            canonicalize({"score": 0.87})

    def test_float_nested_in_list_raises(self):
        with self.assertRaises(CanonicalizationError):
            canonicalize([1, 2, 0.5])

    def test_out_of_range_integer_raises(self):
        with self.assertRaises(CanonicalizationError):
            canonicalize(2**53)


class TestCanonicalizeObjectOrdering(unittest.TestCase):

    def test_keys_sorted(self):
        self.assertEqual(canonicalize({"b": 1, "a": 2}), b'{"a":2,"b":1}')

    def test_nested_object_keys_sorted(self):
        self.assertEqual(
            canonicalize({"z": {"y": 1, "x": 2}}),
            b'{"z":{"x":2,"y":1}}',
        )

    def test_array_order_preserved(self):
        self.assertEqual(canonicalize([3, 1, 2]), b"[3,1,2]")

    def test_empty_object_and_array(self):
        self.assertEqual(canonicalize({"a": {}, "b": []}), b'{"a":{},"b":[]}')

    def test_no_spaces_in_output(self):
        out = canonicalize({"a": [1, 2], "b": "x"})
        self.assertNotIn(b" ", out)


if __name__ == "__main__":
    unittest.main()
