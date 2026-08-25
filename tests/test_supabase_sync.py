import unittest

from supabase_sync import normalize_observation_record


class SupabaseSyncTests(unittest.TestCase):
    def test_normalize_observation_record_extracts_expected_fields(self):
        raw = {
            'id': 12345,
            'uuid': 'abc-123',
            'observed_on': '2026-01-02',
            'taxon': {'name': 'Morchella americana'},
            'geojson': {'coordinates': [-105.2, 40.1]},
            'place_guess': 'Boulder, CO',
            'num_identification_agreements': 7,
        }

        result = normalize_observation_record(raw)

        self.assertEqual(result['inat_id'], 12345)
        self.assertEqual(result['uuid'], 'abc-123')
        self.assertEqual(result['species'], 'Morchella americana')
        self.assertEqual(result['date'], '2026-01-02')
        self.assertEqual(result['lat'], 40.1)
        self.assertEqual(result['lon'], -105.2)
        self.assertEqual(result['location'], 'Boulder, CO')
        self.assertEqual(result['num_identification_agreements'], 7)
        self.assertIn('raw_payload', result)


if __name__ == '__main__':
    unittest.main()
