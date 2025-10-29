# extraction\methods\unittests\test_get_agreements_and_disagreements.py
import unittest

from extraction.methods.get_agreements_and_disagreements import deduplicate_taxa_list_on_scientific_name
from extraction.methods.structured_output_schema import Taxon, TaxaData


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

if __name__ == '__main__':
    unittest.main()