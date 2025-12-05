import unittest

from phytochemMiner import abbreviate_sci_name, check_compound_names_match, check_organism_names_match


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


class TestCheckCompoundNamesMatch(unittest.TestCase):

    def test_exact_match(self):
        self.assertTrue(check_compound_names_match("CompoundA", "CompoundA"))

    def test_case_insensitive_match(self):
        self.assertTrue(check_compound_names_match("compoundA", "CompoundA"))

    def test_whitespace_insensitive_match(self):
        self.assertTrue(check_compound_names_match("  CompoundA ", "CompoundA"))

    def test_normalized_match(self):
        self.assertTrue(check_compound_names_match("Compound   A", "Compound A"))

    def test_mismatch(self):
        self.assertFalse(check_compound_names_match("CompoundA", "CompoundB"))

    def test_one_empty_string(self):
        self.assertFalse(check_compound_names_match("", "CompoundA"))

    def test_both_empty_strings(self):
        self.assertTrue(check_compound_names_match("", ""))


class TestCheckOrganismNamesMatch(unittest.TestCase):

    def test_exact_match(self):
        self.assertTrue(check_organism_names_match("Ficus elastica", "Ficus elastica"))

    def test_abbreviation_match_name1_to_name2(self):
        self.assertTrue(check_organism_names_match("F. elastica", "Ficus elastica"))

    def test_abbreviation_match_name2_to_name1(self):
        self.assertTrue(check_organism_names_match("Ficus elastica", "F. elastica"))

    def test_mismatch(self):
        self.assertFalse(check_organism_names_match("Ficus religiosa", "Ficus elastica"))

    def test_empty_strings(self):
        self.assertTrue(check_organism_names_match("", ""))

    def test_one_empty_string(self):
        self.assertFalse(check_organism_names_match("Ficus elastica", ""))

    def test_hybrid_character_match(self):
        self.assertTrue(check_organism_names_match("× F. elastica", "× Ficus elastica"))

    def test_hybrid_character_mismatch(self):
        self.assertFalse(check_organism_names_match("× Ficus religiosa", "Ficus elastica"))


if __name__ == "__main__":
    unittest.main()
