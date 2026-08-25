import os
import tempfile
import unittest
from unittest import mock

from cluster import stage_output_path
from export_geojson import build_parser
from fetch import fetch_chirps_precip
from iNat import (
    fetch_inat_data,
    format_observation_progress,
    get_elevation,
    parse_species_list,
    should_refresh_all,
    filter_new_observations,
)
from terrain_pipeline import _find_dem


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


class ChirpsDownloadCleanupTests(unittest.TestCase):
    def test_missing_chirps_file_cleans_stale_downloads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            date_str = '2024-05-17'
            out_path = os.path.join(tmpdir, f'precip_{date_str}.tif')
            gz_path = out_path + '.gz'

            with open(gz_path, 'wb') as fh:
                fh.write(b'stale-data')

            response = mock.Mock(status_code=404)
            with mock.patch('fetch.requests.get', return_value=response):
                result = fetch_chirps_precip(date_str, output_dir=tmpdir)

            self.assertIsNone(result)
            self.assertFalse(os.path.exists(out_path))
            self.assertFalse(os.path.exists(gz_path))


class FetchCacheTests(unittest.TestCase):
    def test_fetch_inat_data_skips_existing_ids_before_expensive_enrichment(self):
        payload = {
            'results': [
                {'id': 1, 'uuid': 'a', 'observed_on': '2024-05-17', 'geojson': {'coordinates': [-105.0, 40.0]}, 'taxon': {'name': 'Morchella'}, 'place_guess': 'Boulder', 'num_identification_agreements': 2},
                {'id': 2, 'uuid': 'b', 'observed_on': '2024-05-18', 'geojson': {'coordinates': [-105.1, 40.1]}, 'taxon': {'name': 'Morchella'}, 'place_guess': 'Denver', 'num_identification_agreements': 5},
            ]
        }

        with mock.patch('iNat.get_observations', return_value=payload), \
             mock.patch('iNat.get_elevation', return_value=1500) as mock_elevation, \
             mock.patch('iNat.get_weather', return_value={'tavg': 18.0}) as mock_weather:
            df = fetch_inat_data(
                taxon_name='morchella',
                lat=40.0,
                lng=-105.0,
                radius=500,
                per_page=100,
                existing_ids={'1'},
            )

        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['inat_id'], 2)
        self.assertEqual(mock_elevation.call_count, 1)
        self.assertEqual(mock_weather.call_count, 1)

    def test_get_elevation_uses_cache_for_identical_coordinates(self):
        with mock.patch('requests.get', return_value=mock.Mock(ok=True, json=lambda: {'results': [{'elevation': 123}]})) as mock_get:
            self.assertEqual(get_elevation(40.0, -105.0), 123)
            self.assertEqual(get_elevation(40.0, -105.0), 123)
            self.assertEqual(mock_get.call_count, 1)

    def test_find_dem_falls_back_to_downloaded_file_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dem_path = os.path.join(tmpdir, 'dem_srtmgl3.tif')
            with open(dem_path, 'wb') as fh:
                fh.write(b'test')
            found = _find_dem(tmpdir)
            self.assertEqual(found, dem_path)


if __name__ == '__main__':
    unittest.main()
