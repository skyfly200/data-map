"""Tests for the Earth Engine enrichment stages.

Earth Engine needs a credential and a network, so these drive ``ee_enrich``
against a fake ``ee`` module. That still exercises the parts most likely to
break: how rows are grouped into requests, which columns each stage fills, the
kelvin→celsius conversion, resume behaviour, and the fact that a failed batch
leaves data alone rather than writing garbage.
"""
import sys
import types
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import ee_enrich
import fetch


# ─── A minimal fake Earth Engine ──────────────────────────────────────────────
# Images are symbolic: they record the bands they carry so a reduceRegions can
# answer with one value per band. Values come from a per-test lookup keyed by
# band name, so a stage's column mapping is what is actually under test.

class FakeImage:
    def __init__(self, bands, values):
        self.bands = list(bands)
        self.values = values

    def rename(self, names):
        names = [names] if isinstance(names, str) else list(names)
        return FakeImage(names, self.values)

    def select(self, bands):
        bands = [bands] if isinstance(bands, str) else list(bands)
        return FakeImage(bands, self.values)

    def addBands(self, other):
        return FakeImage(self.bands + other.bands, self.values)

    def subtract(self, other):
        return FakeImage(self.bands, self.values)

    def add(self, other):
        return FakeImage(self.bands, self.values)

    def divide(self, n):
        return FakeImage(self.bands, self.values)

    def resample(self, method):
        return self

    def reproject(self, **kwargs):
        return self

    def reduceNeighborhood(self, **kwargs):
        return self

    def selfMask(self):
        return self

    def clip(self, region):
        return self

    def normalizedDifference(self, bands):
        return FakeImage(['nd'], self.values)

    def reduceRegions(self, collection=None, reducer=None, scale=None):
        features = []
        for feat in collection.features:
            props = dict(feat.props)
            _kind, (lon, lat) = feat.geometry
            for band in self.bands:
                value = self.values.get(band)
                # A callable band value lets a test vary the sample by location,
                # which is what makes the derived terrain indices meaningful.
                if callable(value):
                    value = value(lon, lat)
                if value is not None:
                    props[band] = value
            features.append({'properties': props})
        return FakeResult({'features': features})


class FakeResult:
    def __init__(self, payload):
        self.payload = payload

    def getInfo(self):
        return self.payload


class FakeFeature:
    def __init__(self, geometry, props):
        self.geometry = geometry
        self.props = props


class FakeFeatureCollection:
    def __init__(self, features):
        self.features = list(features)


class FakeImageCollection:
    def __init__(self, values, size=1):
        self.values = values
        self._size = size

    def filterDate(self, *a):
        return self

    def filterBounds(self, *a):
        return self

    def filter(self, *a):
        return self

    def select(self, bands):
        bands = [bands] if isinstance(bands, str) else list(bands)
        return FakeImageCollection(self.values, self._size)

    def map(self, fn):
        return self

    def median(self):
        return FakeImage(['ndvi'], self.values)

    def first(self):
        return FakeImage(['Map'], self.values)

    def size(self):
        return FakeSize(self._size)


class FakeSize:
    def __init__(self, n):
        self.n = n

    def gt(self, other):
        return self.n > other


def make_fake_ee(values, collection_size=1):
    """Build a stand-in ``ee`` module whose samples return ``values`` per band."""
    ee = types.ModuleType('ee')

    ee.Image = lambda arg=None, *a, **k: (
        arg if isinstance(arg, FakeImage) else FakeImage([], values)
    )
    ee.Image.cat = lambda images: FakeImage(
        [b for img in images for b in img.bands], values
    )
    ee.Image.constant = lambda v: FakeImage(['constant'], values)
    ee.ImageCollection = lambda arg=None: (
        arg if isinstance(arg, FakeImageCollection) else FakeImageCollection(values, collection_size)
    )
    ee.Feature = lambda geometry, props: FakeFeature(geometry, props)
    ee.FeatureCollection = lambda feats: FakeFeatureCollection(feats)
    ee.Geometry = types.SimpleNamespace(
        Point=lambda coords: ('point', tuple(coords)),
        Rectangle=lambda coords: ('rect', tuple(coords)),
    )
    ee.Reducer = types.SimpleNamespace(first=lambda: 'first', mean=lambda: 'mean')
    ee.Filter = types.SimpleNamespace(lt=lambda *a: 'lt')
    ee.Kernel = types.SimpleNamespace(circle=lambda r, units: ('circle', r, units))
    ee.Terrain = types.SimpleNamespace(
        products=lambda dem: FakeImage(['elevation', 'slope', 'aspect'], values)
    )
    ee.Date = lambda s: types.SimpleNamespace(advance=lambda n, unit: s)
    ee.Algorithms = types.SimpleNamespace(If=lambda cond, a, b: a if cond else b)
    return ee


def observations(n=3, date='2024-06-01'):
    return pd.DataFrame({
        'uuid': [f'u{i}' for i in range(n)],
        'species': ['Morchella esculenta'] * n,
        'lat': [40.0 + i * 0.01 for i in range(n)],
        'lon': [-105.0 - i * 0.01 for i in range(n)],
        'date': [date] * n,
    })


class _EEStageTest(unittest.TestCase):
    """Base that patches ee_enrich's module-level Earth Engine handle."""

    def setUp(self):
        ee_enrich._ee = None
        self.addCleanup(setattr, ee_enrich, '_ee', None)

    def run_stage(self, fn, df, values, collection_size=1, **kwargs):
        fake = make_fake_ee(values, collection_size)
        with mock.patch.object(ee_enrich, 'init_ee', return_value=fake):
            return fn(df, **kwargs)


class TemperatureStageTests(_EEStageTest):
    def test_kelvin_is_converted_to_celsius(self):
        df = observations(2)
        values = {}
        for d in range(7):
            values[f'tmax_d{d}'] = 293.15   # 20 °C
            values[f'tmin_d{d}'] = 283.15   # 10 °C

        out = self.run_stage(ee_enrich.enrich_temperature_ee, df, values)

        for d in range(7):
            self.assertAlmostEqual(out[f'tmax_d{d}'].iloc[0], 20.0, places=6)
            self.assertAlmostEqual(out[f'tmin_d{d}'].iloc[0], 10.0, places=6)

    def test_every_day_of_the_window_is_requested_in_one_batch(self):
        # Seven days of highs and lows must ride on a single image, so one
        # observation date costs one request rather than fourteen.
        df = observations(2)
        captured = {}
        values = {f'tmax_d{d}': 300.0 for d in range(7)}
        values.update({f'tmin_d{d}': 290.0 for d in range(7)})

        real_sample = ee_enrich._sample_points

        def spy(ee, image, points, scale, reducer=None):
            captured['bands'] = list(image.bands)
            captured['calls'] = captured.get('calls', 0) + 1
            return real_sample(ee, image, points, scale, reducer)

        with mock.patch.object(ee_enrich, '_sample_points', spy):
            self.run_stage(ee_enrich.enrich_temperature_ee, df, values)

        self.assertEqual(captured['calls'], 1)
        self.assertEqual(len(captured['bands']), 14)

    def test_completed_rows_are_skipped(self):
        df = observations(2)
        for d in range(7):
            df[f'tmax_d{d}'] = 1.0
            df[f'tmin_d{d}'] = 0.0

        with mock.patch.object(ee_enrich, '_sample_points') as sampler:
            out = self.run_stage(ee_enrich.enrich_temperature_ee, df,
                                 {'tmax_d0': 300.0})

        sampler.assert_not_called()
        self.assertEqual(out['tmax_d0'].iloc[0], 1.0)


class PrecipitationStageTests(_EEStageTest):
    def test_seven_days_land_in_their_own_columns(self):
        df = observations(2)
        values = {f'prcp_d{d}': float(d) for d in range(7)}

        out = self.run_stage(ee_enrich.enrich_precip_ee, df, values)

        for d in range(7):
            self.assertEqual(out[f'prcp_d{d}'].iloc[0], float(d))
            self.assertEqual(out[f'prcp_d{d}'].iloc[1], float(d))

    def test_one_request_per_observation_date(self):
        df = pd.concat([observations(2, '2024-06-01'), observations(2, '2024-06-05')],
                       ignore_index=True)
        values = {f'prcp_d{d}': 1.0 for d in range(7)}
        calls = []

        real_sample = ee_enrich._sample_points

        def spy(ee, image, points, scale, reducer=None):
            calls.append(len(points))
            return real_sample(ee, image, points, scale, reducer)

        with mock.patch.object(ee_enrich, '_sample_points', spy):
            self.run_stage(ee_enrich.enrich_precip_ee, df, values)

        self.assertEqual(len(calls), 2)     # two dates, two requests
        self.assertEqual(sorted(calls), [2, 2])

    def test_a_missing_day_stays_null_rather_than_zero(self):
        # A gap in CHIRPS must not read as "no rain fell".
        df = observations(1)
        values = {f'prcp_d{d}': 3.0 for d in range(7)}
        del values['prcp_d3']

        out = self.run_stage(ee_enrich.enrich_precip_ee, df, values)

        self.assertEqual(out['prcp_d2'].iloc[0], 3.0)
        self.assertTrue(pd.isna(out['prcp_d3'].iloc[0]))


class SoilMoistureStageTests(_EEStageTest):
    def test_value_is_written_to_the_soil_moisture_column(self):
        df = observations(3)
        out = self.run_stage(ee_enrich.enrich_soil_moisture_ee, df,
                             {'soil_moisture': 0.27})
        self.assertTrue((out['soil_moisture'] == 0.27).all())

    def test_a_failed_batch_leaves_the_column_empty(self):
        df = observations(2)
        with mock.patch.object(ee_enrich, '_sample_points', side_effect=RuntimeError('EE down')):
            out = self.run_stage(ee_enrich.enrich_soil_moisture_ee, df,
                                 {'soil_moisture': 0.27})
        self.assertTrue(out['soil_moisture'].isna().all())


class LandCoverStageTests(_EEStageTest):
    def test_class_code_is_sampled_with_the_first_reducer(self):
        # Averaging class codes would invent classes, so the reducer matters.
        df = observations(2)
        captured = {}
        real_sample = ee_enrich._sample_points

        def spy(ee, image, points, scale, reducer=None):
            captured['reducer'] = reducer
            captured['scale'] = scale
            return real_sample(ee, image, points, scale, reducer)

        with mock.patch.object(ee_enrich, '_sample_points', spy):
            out = self.run_stage(ee_enrich.enrich_landcover_ee, df, {'land_cover': 10})

        self.assertIsNone(captured['reducer'])   # defaults to Reducer.first()
        self.assertEqual(captured['scale'], ee_enrich.SCALE_WORLDCOVER)
        self.assertTrue((out['land_cover'] == 10).all())


class TerrainStageTests(_EEStageTest):
    def test_samples_populate_terrain_and_derived_indices(self):
        df = observations(4)
        values = {'elevation': 2500.0, 'slope': 12.0, 'aspect': 180.0,
                  'tpi': 3.0, 'upa': 0.5}

        out = self.run_stage(ee_enrich.enrich_terrain_ee, df, values)

        self.assertTrue((out['slope'] == 12.0).all())
        self.assertTrue((out['aspect'] == 180.0).all())
        self.assertTrue((out['elevation'] == 2500.0).all())
        for col in ('solar_exposure', 'wind_exposure', 'water_retention'):
            self.assertFalse(out[col].isna().any(), f'{col} was not derived')
            self.assertTrue(((out[col] >= 0) & (out[col] <= 1)).all(),
                            f'{col} outside 0..1')

    def test_wind_exposure_rises_with_topographic_position(self):
        # Openness is 70% of the wind index, so a point on a ridge (high TPI)
        # must come out more exposed than one in a valley (low TPI).
        df = observations(3)
        lat_of = {row.lat: i for i, row in enumerate(df.itertuples())}
        tpi_by_row = [-20.0, 0.0, 25.0]

        values = {
            'elevation': 2500.0, 'slope': 15.0, 'aspect': 180.0, 'upa': 0.5,
            'tpi': lambda lon, lat: tpi_by_row[lat_of[lat]],
        }
        out = self.run_stage(ee_enrich.enrich_terrain_ee, df, values)

        wind = list(out['wind_exposure'])
        self.assertLess(wind[0], wind[1])
        self.assertLess(wind[1], wind[2])

    def test_water_retention_falls_as_slope_steepens(self):
        # TWI = ln(a / tan(slope)): with a fixed contributing area, steeper
        # ground sheds water and must score drier.
        df = observations(3)
        lat_of = {row.lat: i for i, row in enumerate(df.itertuples())}
        slopes = [2.0, 15.0, 40.0]

        values = {
            'elevation': 2500.0, 'aspect': 180.0, 'tpi': 1.0, 'upa': 1.0,
            'slope': lambda lon, lat: slopes[lat_of[lat]],
        }
        out = self.run_stage(ee_enrich.enrich_terrain_ee, df, values)

        water = list(out['water_retention'])
        self.assertGreater(water[0], water[1])
        self.assertGreater(water[1], water[2])

    def test_water_retention_rises_with_upstream_area(self):
        # More upslope drainage converging on a point means a wetter site.
        df = observations(3)
        lat_of = {row.lat: i for i, row in enumerate(df.itertuples())}
        upa = [0.01, 1.0, 100.0]

        values = {
            'elevation': 2500.0, 'aspect': 180.0, 'tpi': 1.0, 'slope': 10.0,
            'upa': lambda lon, lat: upa[lat_of[lat]],
        }
        out = self.run_stage(ee_enrich.enrich_terrain_ee, df, values)

        water = list(out['water_retention'])
        self.assertLess(water[0], water[1])
        self.assertLess(water[1], water[2])

    def test_solar_exposure_favours_south_facing_slopes(self):
        # At northern mid-latitudes a south-facing slope (aspect 180°) receives
        # more potential radiation than the north-facing slope opposite it.
        df = observations(2)
        lat_of = {row.lat: i for i, row in enumerate(df.itertuples())}
        aspects = [0.0, 180.0]   # north-facing, south-facing

        values = {
            'elevation': 2500.0, 'slope': 25.0, 'tpi': 1.0, 'upa': 0.5,
            'aspect': lambda lon, lat: aspects[lat_of[lat]],
        }
        out = self.run_stage(ee_enrich.enrich_terrain_ee, df, values)

        north, south = out['solar_exposure'].iloc[0], out['solar_exposure'].iloc[1]
        self.assertGreater(south, north)

    def test_an_existing_elevation_is_not_overwritten(self):
        # iNaturalist already supplies elevation; the DEM should not clobber it.
        df = observations(2)
        df['elevation'] = 1234.0
        out = self.run_stage(ee_enrich.enrich_terrain_ee, df,
                             {'elevation': 2500.0, 'slope': 5.0, 'aspect': 90.0,
                              'tpi': 1.0, 'upa': 0.2})
        self.assertTrue((out['elevation'] == 1234.0).all())

    def test_rows_without_coordinates_are_left_alone(self):
        df = observations(2)
        df.loc[0, 'lat'] = np.nan
        out = self.run_stage(ee_enrich.enrich_terrain_ee, df,
                             {'elevation': 2500.0, 'slope': 5.0, 'aspect': 90.0,
                              'tpi': 1.0, 'upa': 0.2})
        self.assertTrue(pd.isna(out['slope'].iloc[0]))
        self.assertEqual(out['slope'].iloc[1], 5.0)


class NdviStageTests(_EEStageTest):
    def test_ndvi_is_sampled_per_date_group(self):
        df = pd.concat([observations(2, '2024-06-01'), observations(1, '2024-07-01')],
                       ignore_index=True)
        out = self.run_stage(ee_enrich.enrich_ndvi_ee, df, {'ndvi': 0.62})
        self.assertTrue((out['ndvi'] == 0.62).all())

    def test_rows_with_an_ndvi_value_are_not_resampled(self):
        df = observations(2)
        df['ndvi'] = [0.4, None]
        out = self.run_stage(ee_enrich.enrich_ndvi_ee, df, {'ndvi': 0.9})
        self.assertEqual(out['ndvi'].iloc[0], 0.4)
        self.assertEqual(out['ndvi'].iloc[1], 0.9)


class SamplingPrimitiveTests(_EEStageTest):
    def test_points_are_chunked_to_bound_request_size(self):
        fake = make_fake_ee({'band': 1.0})
        points = [(i, -105.0, 40.0) for i in range(ee_enrich.CHUNK_SIZE * 2 + 7)]
        image = FakeImage(['band'], {'band': 1.0})

        with mock.patch.object(image, 'reduceRegions', wraps=image.reduceRegions) as spy:
            out = ee_enrich._sample_points(fake, image, points, 30)

        self.assertEqual(spy.call_count, 3)
        self.assertEqual(len(out), len(points))

    def test_results_are_keyed_by_caller_position_not_frame_index(self):
        # Stages pass positions, so a non-integer or non-contiguous DataFrame
        # index can never mis-map a sampled value onto the wrong row.
        fake = make_fake_ee({'band': 5.0})
        image = FakeImage(['band'], {'band': 5.0})
        out = ee_enrich._sample_points(fake, image, [(0, -105.0, 40.0), (1, -104.0, 41.0)], 30)
        self.assertEqual(out, {0: {'band': 5.0}, 1: {'band': 5.0}})

    def test_non_contiguous_index_maps_values_to_the_right_rows(self):
        df = observations(3)
        df.index = [10, 20, 30]
        out = self.run_stage(ee_enrich.enrich_soil_moisture_ee, df,
                             {'soil_moisture': 0.42})
        self.assertEqual(list(out.index), [10, 20, 30])
        self.assertTrue((out['soil_moisture'] == 0.42).all())


class EarthEngineToggleTests(unittest.TestCase):
    def test_enabled_by_default(self):
        with mock.patch.dict('os.environ', {}, clear=True):
            self.assertTrue(ee_enrich.earth_engine_enabled())

    def test_skip_earth_engine_wins(self):
        with mock.patch.dict('os.environ', {'SKIP_EARTH_ENGINE': '1'}, clear=True):
            self.assertFalse(ee_enrich.earth_engine_enabled())

    def test_use_earth_engine_off(self):
        with mock.patch.dict('os.environ', {'USE_EARTH_ENGINE': '0'}, clear=True):
            self.assertFalse(ee_enrich.earth_engine_enabled())

    def test_init_returns_none_when_disabled(self):
        with mock.patch.dict('os.environ', {'SKIP_EARTH_ENGINE': '1'}, clear=True):
            ee_enrich._ee = None
            self.assertIsNone(ee_enrich.init_ee())


class RasterDownloadSkipTests(unittest.TestCase):
    """fetch.py must only skip the bulk downloads when EE will really cover them."""

    def test_skipped_when_earth_engine_is_available(self):
        self.assertTrue(fetch.skip_raster_downloads(ee_available=True, env={}))

    def test_downloaded_when_earth_engine_is_unavailable(self):
        self.assertFalse(fetch.skip_raster_downloads(ee_available=False, env={}))

    def test_fetch_rasters_forces_the_download(self):
        self.assertFalse(
            fetch.skip_raster_downloads(ee_available=True, env={'FETCH_RASTERS': '1'}))

    def test_skip_earth_engine_forces_the_download(self):
        self.assertFalse(
            fetch.skip_raster_downloads(ee_available=True, env={'SKIP_EARTH_ENGINE': '1'}))

    def test_use_earth_engine_off_forces_the_download(self):
        self.assertFalse(
            fetch.skip_raster_downloads(ee_available=True, env={'USE_EARTH_ENGINE': '0'}))


class RasterFallbackDoesNotClobberTests(unittest.TestCase):
    """The raster stages run after Earth Engine, so they must only fill gaps.

    Cached rasters can be older than the Earth Engine sample (or absent for the
    dates EE covered), so a stage that rewrites rows it did not need to touch
    would quietly replace fresh values with stale ones.
    """

    def test_precip_leaves_rows_that_already_have_rainfall(self):
        import enrich_with_rasters as enrich

        df = observations(2)
        for d in range(7):
            df[f'prcp_d{d}'] = None
        # Row 0 already sampled by Earth Engine; row 1 still empty.
        for d in range(7):
            df.loc[0, f'prcp_d{d}'] = 9.0

        with mock.patch.object(enrich, 'resolve_raster_path', return_value='fake.tif'), \
             mock.patch.object(enrich, 'sample_raster_points', return_value=[0.0]) as sampler:
            out = enrich.enrich_with_precip(df, precip_dir='precip/')

        # Only the one unfilled row was sampled, and the filled row is intact.
        for call in sampler.call_args_list:
            self.assertEqual(len(call.args[1]), 1)
        for d in range(7):
            self.assertEqual(out.loc[0, f'prcp_d{d}'], 9.0)
            self.assertEqual(out.loc[1, f'prcp_d{d}'], 0.0)

    def test_ndvi_soil_pass_skips_dates_that_are_already_complete(self):
        import enrich_with_rasters as enrich

        df = observations(2)
        df['ndvi'] = 0.7
        df['soil_moisture'] = 0.3

        with mock.patch.object(enrich, 'resolve_raster_path') as resolver:
            out = enrich.enrich_df_with_rasters(df)

        resolver.assert_not_called()
        self.assertTrue((out['ndvi'] == 0.7).all())
        self.assertTrue((out['soil_moisture'] == 0.3).all())

    def test_ndvi_soil_pass_fills_only_the_empty_cells(self):
        import enrich_with_rasters as enrich

        df = observations(2)
        df['ndvi'] = [0.7, None]
        df['soil_moisture'] = None

        with mock.patch.object(enrich, 'resolve_raster_path', return_value='ndvi.tif'), \
             mock.patch.object(enrich, 'get_ndvi_from_raster', return_value=0.1), \
             mock.patch('os.path.exists', return_value=False):
            out = enrich.enrich_df_with_rasters(df)

        self.assertEqual(out['ndvi'].iloc[0], 0.7)   # untouched
        self.assertEqual(out['ndvi'].iloc[1], 0.1)   # filled from the raster

    def test_terrain_is_quiet_when_earth_engine_already_filled_it(self):
        # No warning about missing DEM layers when there is nothing left to do.
        import enrich_with_rasters as enrich

        df = observations(2)
        for col in enrich.TERRAIN_LAYERS:
            df[col] = 0.5

        with mock.patch.object(enrich, 'glob') as globber:
            out = enrich.enrich_with_terrain(df, terrain_dir='nonexistent/')

        globber.glob.assert_not_called()
        for col in enrich.TERRAIN_LAYERS:
            self.assertTrue((out[col] == 0.5).all())


if __name__ == '__main__':
    unittest.main()
