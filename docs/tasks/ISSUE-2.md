# ISSUE-2 Large GBIF exports timeout during import

Large GBIF exports (>10k records) may timeout during import. The fix is streaming the response into Supabase Storage incrementally rather than building the full GeoJSON in memory. This is also a prerequisite for `WANT-9` (GBIF API direct loading).
