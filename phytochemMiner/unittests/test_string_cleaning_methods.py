# File: extraction\methods\unittests\test_string_cleaning_methods.py

import unittest

from phytochemMiner import clean_taxon_strings, clean_compound_strings


class TestCleanTaxonStrings(unittest.TestCase):
    def test_basic_cleaning(self):
        given_str = "  Hello, World!!  "
        expected = "hello, world"
        self.assertEqual(expected, clean_taxon_strings(given_str))

    def test_empty_string(self):
        given_str = ""
        expected = ""
        self.assertEqual(expected, clean_taxon_strings(given_str))

    def test_single_word(self):
        given_str = " Taxon ."
        expected = "taxon"
        self.assertEqual(expected, clean_taxon_strings(given_str))

    def test_no_changes_needed(self):
        given_str = "simple example"
        expected = "simple example"
        self.assertEqual(expected, clean_taxon_strings(given_str))

    def test_multiple_whitespace_and_breaks(self):
        given_str = "line1\nline2\t   line3"
        expected = "line1 line2 line3"
        self.assertEqual(expected, clean_taxon_strings(given_str))

    def test_leading_and_trailing_punctuation(self):
        given_str = "!@#TaxonName***---"
        expected = "taxonname"
        self.assertEqual(expected, clean_taxon_strings(given_str))

    def test_mixed_case_handling(self):
        given_str = "MiXeD CaSe VaLuE"
        expected = "mixed case value"
        self.assertEqual(expected, clean_taxon_strings(given_str))

    def test_all_punctuation_string(self):
        given_str = "!!!###@@@$$$"
        expected = ""
        self.assertEqual(expected, clean_taxon_strings(given_str))

    def test_none_input(self):
        given_str = None
        expected = None
        self.assertEqual(expected, clean_taxon_strings(given_str))


class TestCleanCompoundStrings(unittest.TestCase):
    def test_basic_cleaning(self):
        given_str = "  Hello  World  "
        expected = "hello world"
        self.assertEqual(expected, clean_compound_strings(given_str))

    def test_empty_string(self):
        given_str = ""
        expected = ""
        self.assertEqual(expected, clean_compound_strings(given_str))

    def test_clean_multiple_whitespace(self):
        given_str = "  Multi    Whitespace \nText "
        expected = "multi whitespace text"
        self.assertEqual(expected, clean_compound_strings(given_str))

    def test_none_input(self):
        given_str = None
        expected = None
        self.assertEqual(expected, clean_compound_strings(given_str))

    def test_no_changes_needed(self):
        given_str = "already clean"
        expected = "already clean"
        self.assertEqual(expected, clean_compound_strings(given_str))

    def test_leading_and_trailing_spaces(self):
        given_str = "\n leading and trailing spaces \t"
        expected = "leading and trailing spaces"
        self.assertEqual(expected, clean_compound_strings(given_str))

    def test_large_whitespace_and_empty_lines(self):
        given_str = """


        Large       whitespace
        with multiple lines


        """
        expected = "large whitespace with multiple lines"
        self.assertEqual(expected, clean_compound_strings(given_str))


if __name__ == '__main__':
    unittest.main()
