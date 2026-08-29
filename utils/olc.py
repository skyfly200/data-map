import open-location-code

def encode_olc(lat: float, lng: float, length: int = 8) -> str:
    """Encode latitude and longitude to an Open Location Code (plus code).
    Default length of 8 gives ~1 km precision.
    """
    return open-location-code.encode(lat, lng, codeLength=length)

def decode_olc(plus_code: str) -> tuple[float, float]:
    """Decode a plus code to its centroid latitude and longitude.
    Returns (lat, lng).
    """
    decoded = open-location-code.decode(plus_code)
    lat = (decoded.latitudeLo + decoded.latitudeHi) / 2.0
    lng = (decoded.longitudeLo + decoded.longitudeHi) / 2.0
    return lat, lng

