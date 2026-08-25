import unittest

from cluster import stage_output_path
from iNat import parse_species_list


class SpeciesListParsingTests(unittest.TestCase):
    def test_string_list_is_parsed_into_individual_species(self):
        self.assertEqual(parse_species_list('amanita, morchella, boletus'), ['amanita', 'morchella', 'boletus'])

    def test_duplicate_and_blank_entries_are_removed(self):
        self.assertEqual(parse_species_list(' amanita, , morchella, amanita '), ['amanita', 'morchella'])


class UniqueStageNamingTests(unittest.TestCase):
    def test_enriched_stage_uses_unique_input_stem(self):
        self.assertEqual(
            stage_output_path('mushroom_observations_amanita_40.0_-105.0_500.0_20260824T224804Z.csv', '_enriched'),
            'mushroom_observations_amanita_40.0_-105.0_500.0_20260824T224804Z_enriched.csv',
        )

    def test_cluster_stage_uses_unique_input_stem(self):
        self.assertEqual(
            stage_output_path('mushroom_observations_amanita_40.0_-105.0_500.0_20260824T224804Z.csv', '_clusters'),
            'mushroom_observations_amanita_40.0_-105.0_500.0_20260824T224804Z_clusters.csv',
        )


if __name__ == '__main__':
    unittest.main()
