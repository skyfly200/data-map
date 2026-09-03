import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import rasterio

from cluster import stage_output_path
from compress_rasters import convert_raster_to_cog
from enrich_with_rasters import filter_non_productive_landcover, should_filter_non_productive_landcover
from export_geojson import build_parser
from fetch import fetch_chirps_precip
from iNat import (
    fetch_inat_data,
    format_observation_progress,
    get_elevation,
    get_parallel_fetch_workers,
    parse_species_list,
    parse_plus_codes,
    parse_plus_code_ranges,
    resolve_inat_page_size,
    should_refresh_all,
    filter_new_observations,
    _geo_query,
    _retry_after_seconds,
)
from run_pipeline import should_skip_stage
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

    def test_default_page_size_is_higher_than_100(self):
        self.assertEqual(resolve_inat_page_size({}), 200)
        self.assertEqual(resolve_inat_page_size({'INAT_PER_PAGE': '400'}), 400)

    def test_parallel_fetch_workers_are_configurable(self):
        self.assertEqual(get_parallel_fetch_workers({'INAT_PARALLEL_FETCHES': '4'}), 4)
        # Default is serial (1): the shared session rate-limits as a group, so
        # extra workers only add per-IP throttle pressure.
        self.assertEqual(get_parallel_fetch_workers({}), 1)


class PlusCodeAreaTests(unittest.TestCase):
    def test_plus_code_becomes_a_bounding_box_not_a_radius(self):
        # A full plus code decodes to its rectangular cell; the location carries
        # the four corners, never a radius.
        locations = parse_plus_codes('84QW4600+', default_radius=500.0)
        self.assertEqual(len(locations), 1)
        loc = locations[0]
        self.assertEqual(set(loc), {'swlat', 'swlng', 'nelat', 'nelng', 'label'})
        self.assertNotIn('radius', loc)
        self.assertLess(loc['swlat'], loc['nelat'])
        self.assertLess(loc['swlng'], loc['nelng'])
        # The code's own centroid must fall inside its box.
        self.assertTrue(loc['swlat'] <= 45.10 <= loc['nelat'])
        self.assertTrue(loc['swlng'] <= -121.775 <= loc['nelng'])

    def test_trailing_radius_field_is_ignored_for_plus_codes(self):
        # Radius does not apply to a box; an extra field must not change the area.
        with_extra = parse_plus_codes('84QW4600+,50', default_radius=500.0)[0]
        plain = parse_plus_codes('84QW4600+', default_radius=500.0)[0]
        self.assertEqual(with_extra, plain)

    def test_invalid_plus_code_is_skipped(self):
        self.assertEqual(parse_plus_codes('not-a-code', default_radius=500.0), [])

    def test_geo_query_prefers_bounds_over_point_radius(self):
        bounds = (45.10, -121.80, 45.15, -121.75)  # swlat, swlng, nelat, nelng
        self.assertEqual(
            _geo_query(lat=1.0, lng=2.0, radius=50.0, bounds=bounds),
            {'nelat': 45.15, 'nelng': -121.75, 'swlat': 45.10, 'swlng': -121.80},
        )

    def test_geo_query_falls_back_to_point_radius(self):
        self.assertEqual(
            _geo_query(lat=40.0, lng=-105.0, radius=500.0),
            {'lat': 40.0, 'lng': -105.0, 'radius': 500.0},
        )

    def test_plus_code_range_unions_two_cells_into_one_box(self):
        loc = parse_plus_code_ranges('84QWJF00+:84QWQM00+')[0]
        self.assertEqual(set(loc), {'swlat', 'swlng', 'nelat', 'nelng', 'label'})
        # The union must contain both codes' own cells.
        a_sw_lat, a_sw_lng, a_ne_lat, a_ne_lng = __import__('utils.olc', fromlist=['decode_olc_bounds']).decode_olc_bounds('84QWJF00+')
        self.assertLessEqual(loc['swlat'], a_sw_lat)
        self.assertGreaterEqual(loc['nelat'], a_ne_lat)
        self.assertLess(loc['swlat'], loc['nelat'])
        self.assertLess(loc['swlng'], loc['nelng'])

    def test_plus_code_range_is_order_independent(self):
        self.assertEqual(
            parse_plus_code_ranges('84QV0000+ 84QX0000+'),
            parse_plus_code_ranges('84QX0000+ 84QV0000+'),
        )

    def test_plus_code_range_accepts_colon_or_space_separator(self):
        self.assertEqual(
            parse_plus_code_ranges('84QV0000+:84QX0000+'),
            parse_plus_code_ranges('84QV0000+ 84QX0000+'),
        )

    def test_plus_code_range_needs_exactly_two_codes(self):
        self.assertEqual(parse_plus_code_ranges('84QV0000+'), [])
        self.assertEqual(parse_plus_code_ranges(''), [])
        self.assertEqual(parse_plus_code_ranges('84QV0000+ 84QX0000+ 84QW0000+'), [])

    def test_fetch_inat_data_sends_bounding_box_to_the_api(self):
        payload = {'results': []}
        with mock.patch('iNat.get_observations', return_value=payload) as mock_obs:
            fetch_inat_data(taxon_name='morchella',
                            bounds=(45.10, -121.80, 45.15, -121.75),
                            per_page=100)
        _, kwargs = mock_obs.call_args
        self.assertEqual(kwargs['nelat'], 45.15)
        self.assertEqual(kwargs['nelng'], -121.75)
        self.assertEqual(kwargs['swlat'], 45.10)
        self.assertEqual(kwargs['swlng'], -121.80)
        # No circular query params when a box is requested.
        self.assertNotIn('radius', kwargs)
        self.assertNotIn('lat', kwargs)


class ThrottleBackoffTests(unittest.TestCase):
    class _Resp:
        def __init__(self, status, headers=None):
            self.status_code = status
            self.headers = headers or {}

    class _Err(Exception):
        def __init__(self, resp):
            self.response = resp

    def test_429_honours_retry_after_header(self):
        err = self._Err(self._Resp(429, {'Retry-After': '30'}))
        self.assertEqual(_retry_after_seconds(err, 1), 30.0)

    def test_429_without_header_backs_off_exponentially_and_caps(self):
        self.assertEqual(_retry_after_seconds(self._Err(self._Resp(429)), 1), 5.0)
        self.assertEqual(_retry_after_seconds(self._Err(self._Resp(429)), 3), 20.0)
        self.assertEqual(_retry_after_seconds(self._Err(self._Resp(429)), 10), 60.0)

    def test_non_429_uses_short_linear_backoff(self):
        self.assertEqual(_retry_after_seconds(self._Err(self._Resp(500)), 2), 2.0)
        self.assertEqual(_retry_after_seconds(Exception('boom'), 3), 3.0)


class ResumeAndCompressionTests(unittest.TestCase):
    def test_should_skip_stage_uses_existing_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, 'ready.tif')
            with open(out_path, 'wb') as fh:
                fh.write(b'test')
            self.assertTrue(should_skip_stage(out_path))
            self.assertFalse(should_skip_stage(os.path.join(tmpdir, 'missing.tif')))

    def test_convert_raster_to_cog_round_trip_preserves_pixel_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, 'sample.tif')
            arr = np.arange(100, dtype='float32').reshape(10, 10)
            with rasterio.open(
                src_path,
                'w',
                driver='GTiff',
                width=10,
                height=10,
                count=1,
                dtype='float32',
            ) as dst:
                dst.write(arr, 1)

            converted = convert_raster_to_cog(src_path, delete_original=False, verify=True)
            with rasterio.open(converted) as src:
                converted_arr = src.read(1)

            self.assertTrue(np.allclose(converted_arr, arr, equal_nan=True))
            self.assertTrue(os.path.exists(converted))


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

    def test_non_productive_landcover_rows_are_filtered(self):
        df = pd.DataFrame({
            'land_cover': [10, 50, 70, 80, 90, None],
            'land_cover_label': ['Tree cover', 'Built-up', 'Snow and ice', 'Water', 'Wetland', 'Unknown'],
        })
        filtered = filter_non_productive_landcover(df)
        self.assertEqual(list(filtered['land_cover'].fillna(-1)), [10.0, 90.0, -1])

    def test_filtering_can_be_disabled_via_env_toggle(self):
        self.assertFalse(should_filter_non_productive_landcover({'FILTER_NON_PRODUCTIVE_LANDCOVER': '0'}))
        self.assertTrue(should_filter_non_productive_landcover({'FILTER_NON_PRODUCTIVE_LANDCOVER': '1'}))


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
