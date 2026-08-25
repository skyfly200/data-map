import unittest

from cluster import stage_output_path
from export_geojson import build_parser
from iNat import format_observation_progress, parse_species_list, should_refresh_all, filter_new_observations


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


class GeojsonCliCompatibilityTests(unittest.TestCase):
    def test_output_arg_is_accepted(self):
        parser = build_parser()
        args = parser.parse_args(['--input', 'mushroom_observations.csv', '--output', 'public/data/custom.geojson'])
        self.assertEqual(args.output, 'public/data/custom.geojson')

    def test_data_dir_arg_is_accepted(self):
        parser = build_parser()
        args = parser.parse_args(['--input', 'mushroom_observations.csv', '--data-dir', 'public/data'])
        self.assertEqual(args.data_dir, 'public/data')


class ObservationProgressTests(unittest.TestCase):
    def test_format_observation_progress_includes_species_and_counts(self):
        self.assertEqual(
            format_observation_progress('morchella', 2, 5),
            'morchella [########------------] 2/5',
        )


class IncrementalRefreshTests(unittest.TestCase):
    def test_refresh_all_flag_defaults_to_false(self):
        self.assertFalse(should_refresh_all({'REFRESH_ALL': ''}))
        self.assertFalse(should_refresh_all({}))

    def test_refresh_all_flag_can_be_enabled(self):
        self.assertTrue(should_refresh_all({'REFRESH_ALL': '1'}))
        self.assertTrue(should_refresh_all({'INAT_REFRESH_ALL': 'true'}))

    def test_only_new_observations_are_kept(self):
        existing = [{'inat_id': 1}, {'inat_id': 2}]
        fresh = [{'inat_id': 1}, {'inat_id': 3}, {'inat_id': 4}]
        self.assertEqual(
            [item['inat_id'] for item in filter_new_observations(fresh, existing)],
            [3, 4],
        )


if __name__ == '__main__':
    unittest.main()
