import unittest

from extraction.methods.string_matching_methods import abbreviate_sci_name


class TestAbbreviateSciName(unittest.TestCase):

    def test_single_word_name(self):
        self.assertEqual("Ficus", abbreviate_sci_name("Ficus"))

    def test_two_word_name(self):
        self.assertEqual("F. elastica", abbreviate_sci_name("Ficus elastica"))

    def test_hybrid_character_in_name_short(self):
        self.assertEqual("× Ficus", abbreviate_sci_name("× Ficus"))

    def test_hybrid_character_in_name_long(self):
        self.assertEqual("× F. elastica", abbreviate_sci_name("× Ficus elastica"))

    def test_three_word_name(self):
        self.assertEqual("F. elastica robusta", abbreviate_sci_name("Ficus elastica robusta"))

    def test_empty_string(self):
        self.assertEqual("", abbreviate_sci_name(""))

    def test_space_only_input(self):
        self.assertEqual("  ", abbreviate_sci_name("  "))

    def test_name_with_no_split_characters(self):
        self.assertEqual("HybridPlantName", abbreviate_sci_name("HybridPlantName"))


if __name__ == "__main__":
    unittest.main()
