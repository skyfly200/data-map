import os
import utils.openlocationcode as olc

def get_plus_code_length():
    """Get the plus code length from environment variable or default to 6."""
    try:
        value = os.getenv('PLUS_CODE_LENGTH')
        if value is not None and value.strip() != '':
            length = int(value.strip())
            # Ensure length is within valid range (2-15 for Open Location Codes)
            return max(2, min(15, length))
    except (ValueError, TypeError):
        pass
    return 6

def encode_olc(lat: float, lng: float, length: int = None) -> str:
    """Encode latitude and longitude to an Open Location Code (plus code).
    Default length of 6 gives ~20 km precision for broader regional grouping.
    Can be overridden with PLUS_CODE_LENGTH environment variable.
    """
    if length is None:
        length = get_plus_code_length()
    return olc.encode(lat, lng, codeLength=length)

def decode_olc(plus_code: str) -> tuple[float, float]:
    """Decode a plus code to its centroid latitude and longitude.
    Returns (lat, lng).
    """
    decoded = olc.decode(plus_code)
    lat = (decoded.latitudeLo + decoded.latitudeHi) / 2.0
    lng = (decoded.longitudeLo + decoded.longitudeHi) / 2.0
    return lat, lng

