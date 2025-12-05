import unittest

from phytochemMiner import resolve_name_to_inchi


class TestResolveNameToInchi(unittest.TestCase):

    def test_name_resolved_by_cirpy(self):
        self.assertEqual("WQZGKKKJIJFFOK-GASJEMHNSA-N", resolve_name_to_inchi("glucose"))
        self.assertEqual("WQZGKKKJIJFFOK-GASJEMHNSA-N", resolve_name_to_inchi("Glucose"))
        self.assertEqual("QEVHRUUCFGRFIF-MDEJGZGSSA-N", resolve_name_to_inchi("Reserpine"))
        self.assertEqual("QEVHRUUCFGRFIF-MDEJGZGSSA-N", resolve_name_to_inchi("  Reserpine"))
        self.assertEqual("QEVHRUUCFGRFIF-MDEJGZGSSA-N", resolve_name_to_inchi("reserpine"))
        self.assertEqual("PFTAWBLQPZVEMU-DZGCQCFKSA-N", resolve_name_to_inchi(" Catechin  "))
        self.assertEqual("PFTAWBLQPZVEMU-DZGCQCFKSA-N", resolve_name_to_inchi(
            " (2R,3S)-2-(3,4-dihydroxyphenyl)-3,4-dihydro-2H-chromene-3,5,7-triol  "))

    def test_none_examples(self):
        self.assertIsNone(resolve_name_to_inchi(""))

        self.assertIsNone(resolve_name_to_inchi("surelythiscantbeacompound"))
        self.assertIsNone(resolve_name_to_inchi(None))


if __name__ == "__main__":
    unittest.main()
