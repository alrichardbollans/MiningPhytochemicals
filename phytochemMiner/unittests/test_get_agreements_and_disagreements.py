# extraction\methods\unittests\test_get_agreements_and_disagreements.py
import unittest

from analysis.dataset_summaries.get_agreements_and_disagreements import deduplicate_taxa_list_on_scientific_name, \
    deduplicate_taxa_list_on_accepted_name
from phytochemMiner import Taxon, TaxaData


class TestDeduplicateTaxaListOnScientificName(unittest.TestCase):
    def test_no_duplicates(self):
        taxa = [
            Taxon(scientific_name="Ficus religiosa", compounds=["compound1"], inchi_key_simps=[], accepted_name='a'),
            Taxon(scientific_name="Mangifera indica", compounds=["compound2"], inchi_key_simps=[], accepted_name='a')
        ]
        taxadat = TaxaData(taxa=taxa)
        result = deduplicate_taxa_list_on_scientific_name(taxadat)
        expected_taxa = [
            Taxon(scientific_name="Ficus religiosa", compounds=["compound1"], inchi_key_simps=[], accepted_name='a'),
            Taxon(scientific_name="Mangifera indica", compounds=["compound2"], inchi_key_simps=[], accepted_name='b')
        ]
        self.assertEqual(len(expected_taxa), len(result.taxa))
        for i, taxon in enumerate(result.taxa):
            self.assertEqual(expected_taxa[i].scientific_name, taxon.scientific_name)

    def test_with_duplicates(self):
        taxa = [
            Taxon(scientific_name="Ficus religiosa", compounds=["compound1"], inchi_key_simps=[], accepted_name='a'),
            Taxon(scientific_name="Ficus religiosa", compounds=["compound2"], inchi_key_simps=[], accepted_name='a'),
            Taxon(scientific_name="Mangifera indica", compounds=["compound3"], inchi_key_simps=[], accepted_name='a')
        ]
        taxadat = TaxaData(taxa=taxa)
        result = deduplicate_taxa_list_on_scientific_name(taxadat)
        expected_taxa = [
            Taxon(scientific_name="Ficus religiosa", compounds=["compound1", "compound2"], inchi_key_simps=[],
                  accepted_name=None),
            Taxon(scientific_name="Mangifera indica", compounds=["compound3"], inchi_key_simps=[], accepted_name=None)
        ]
        self.assertEqual(len(expected_taxa), len(result.taxa))
        for i, taxon in enumerate(result.taxa):
            self.assertEqual(expected_taxa[i].scientific_name, taxon.scientific_name)

    def test_empty_taxa_list(self):
        taxadat = TaxaData(taxa=[])
        result = deduplicate_taxa_list_on_scientific_name(taxadat)
        self.assertEqual(0, len(result.taxa))

    def test_none_scientific_name(self):
        taxa = [
            Taxon(scientific_name=None, compounds=["compound1"], inchi_key_simps=[], accepted_name='a'),
            Taxon(scientific_name="Ficus religiosa", compounds=["compound2"], inchi_key_simps=[], accepted_name='a'),
            Taxon(scientific_name="Ficus religiosa", compounds=["compound3"], inchi_key_simps=[], accepted_name='a'),
        ]
        taxadat = TaxaData(taxa=taxa)
        result = deduplicate_taxa_list_on_scientific_name(taxadat)
        expected_taxa = [
            Taxon(scientific_name="Ficus religiosa", compounds=["compound2", "compound3"], inchi_key_simps=[],
                  accepted_name=None)
        ]
        self.assertEqual(len(expected_taxa), len(result.taxa))
        for i, taxon in enumerate(result.taxa):
            self.assertEqual(expected_taxa[i].scientific_name, taxon.scientific_name)


class TestDeduplicateTaxaListOnAcceptedName(unittest.TestCase):
    def test_no_duplicates(self):
        taxa = [
            Taxon(scientific_name="Ficus religiosa", compounds=[], inchi_key_simps=["ikey1"], accepted_name="name1"),
            Taxon(scientific_name="Ficus religiosa", compounds=[], inchi_key_simps=["ikey2"], accepted_name="name2")
        ]
        taxadat = TaxaData(taxa=taxa)
        result = deduplicate_taxa_list_on_accepted_name(taxadat)
        expected_taxa = [
            Taxon(scientific_name=None, compounds=[], inchi_key_simps=["ikey1"], accepted_name="name1"),
            Taxon(scientific_name=None, compounds=[], inchi_key_simps=["ikey2"], accepted_name="name2")
        ]
        self.assertEqual(len(expected_taxa), len(result.taxa))
        for i, taxon in enumerate(result.taxa):
            self.assertEqual(expected_taxa[i].accepted_name, taxon.accepted_name)

    def test_with_duplicates(self):
        taxa = [
            Taxon(scientific_name="PlantA", compounds=["cmp1"], inchi_key_simps=["key1"], accepted_name="AccName1"),
            Taxon(scientific_name="PlantB", compounds=["cmp2"], inchi_key_simps=["key2"], accepted_name="AccName1"),
            Taxon(scientific_name="PlantC", compounds=["cmp3"], inchi_key_simps=["key3"], accepted_name="AccName2"),
        ]
        taxadat = TaxaData(taxa=taxa)
        result = deduplicate_taxa_list_on_accepted_name(taxadat)
        expected_taxa = [
            Taxon(scientific_name=None, compounds=["cmp1", "cmp2"], inchi_key_simps=["key1", "key2"],
                  accepted_name="AccName1"),
            Taxon(scientific_name=None, compounds=["cmp3"], inchi_key_simps=["key3"], accepted_name="AccName2"),
        ]
        self.assertEqual(len(expected_taxa), len(result.taxa))
        for i, taxon in enumerate(result.taxa):
            self.assertEqual(expected_taxa[i].accepted_name, taxon.accepted_name)

    def test_empty_taxa_list(self):
        taxadat = TaxaData(taxa=[])
        result = deduplicate_taxa_list_on_accepted_name(taxadat)
        self.assertEqual(0, len(result.taxa))

    def test_none_accepted_name(self):
        taxa = [
            Taxon(scientific_name="PlantX", compounds=[], inchi_key_simps=[], accepted_name=None),
            Taxon(scientific_name="PlantY", compounds=["cmp_a"], inchi_key_simps=["key_a"], accepted_name="AccNameA"),
            Taxon(scientific_name="PlantZ", compounds=["cmp_b"], inchi_key_simps=["key_b"], accepted_name="AccNameA"),
        ]
        taxadat = TaxaData(taxa=taxa)
        result = deduplicate_taxa_list_on_accepted_name(taxadat)
        expected_taxa = [
            Taxon(scientific_name=None, compounds=["cmp_a", "cmp_b"], inchi_key_simps=["key_a", "key_b"],
                  accepted_name="AccNameA")
        ]
        self.assertEqual(len(expected_taxa), len(result.taxa))
        for i, taxon in enumerate(result.taxa):
            self.assertEqual(expected_taxa[i].accepted_name, taxon.accepted_name)


if __name__ == '__main__':
    unittest.main()
