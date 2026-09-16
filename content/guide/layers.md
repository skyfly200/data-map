# Your own Earth Engine layers

An administrator can add map layers that are computed in Earth Engine. They draw
next to the built-in layers.

1. Build the layer in the [Earth Engine Code
   Editor](https://code.earthengine.google.com/).
2. Export it with `Export.image.toAsset()` into your own Earth Engine project.
3. Give the service account of the app read access to the asset.
4. Register it on **Administration → Map layers**. Enter the asset ID, the band
   to draw, a palette, and who can see it.

The layer then appears in the layer window of the map, in the group that you
name. It has its own key, and it renders through the same path as each built-in
layer.

## How the asset and the token fit together

The asset is permanent. The tile URL is not. The app is built around that
difference.

![The asset does not expire. The tile URL does, so the app asks for a new one about every hour.](figure:asset-and-token)

`Export.image.toAsset()` writes your processed grid into a managed record in
your own Earth Engine project. It stays there and it does not expire.

But Earth Engine does not serve raw tiles publicly. A call to `getMapId()` on
the asset creates a URL. That URL carries a token, and the token expires within
hours.

A function on the server is therefore the bridge:

1. The browser asks the function for a tile URL.
2. The function signs in with the service account.
3. It loads the asset that is already computed. It does not repeat the `remap()`
   and `visualize()` steps that the export did.
4. It calls `getMapId()` and returns the new URL.
5. Leaflet draws the tiles.

Nothing is cut into thousands of stored `z/x/y.png` files. There is one asset in
Earth Engine, and one small function that creates tokens on request.

The app asks for a new URL approximately every hour. An expired template does
not give an error. It serves blank tiles, and blank ground reads as ground with
nothing on it, and not as a layer that failed.

> **Note** This is also why the app files the tiles of an Earth Engine layer
> under the layer and not under the URL. Under the URL, each saved tile would be
> unreachable when the token changed.

## A tree cover layer, from start to end

Classify or threshold the data in the Code Editor. Export it once. Register the
asset. The export is what makes the layer cheap to draw, because the work
happens one time and not for each tile.

```js
// Ten cover classes over an area of interest, exported once as an asset.
var aoi = ee.Geometry.Rectangle([-105.4, 39.5, -104.6, 40.2]);

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(aoi)
  .filterDate('2025-06-01', '2025-09-15')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .median();

var ndvi = s2.normalizedDifference(['B8', 'B4']).rename('ndvi');

// Ten classes, 1..10. Zero stays free to mean nodata, which is what the
// layer's "Hide values below" setting masks out.
var classes = ndvi.multiply(10).ceil().clamp(1, 10).toByte().rename('classification');

Export.image.toAsset({
  image: classes.clip(aoi),
  description: 'cover_classes_2025',
  assetId: 'projects/your-project/assets/cover-classes-2025',
  region: aoi,
  scale: 20,
  maxPixels: 1e10
});
```

When the export finishes, give the service account of the app read access to the
asset. Then register the asset on **Administration → Map layers**.

Select the **Classified cover** preset. It fills in the band, the range of 1 to
10, ten different colors and the zero mask. These are the settings that are easy
to get wrong. When they are wrong, the layer renders badly instead of giving an
error.

## The presets

The presets describe the shapes that these rasters usually take: percent cover,
a classified grid, canopy height in metres, a signed index, a probability.

No preset carries an asset ID. A preset describes how to paint a raster. Which
asset you have is your own business.

## Who can see a layer

Each layer carries its own access level: **everyone**, **members** or
**administrators**. A finished layer can therefore be public while a draft stays
internal.

The layer window lists a layer that the viewer cannot render, and marks it. It
does not hide it. When the viewer switches it on, the app explains why it cannot
draw it.

> **Note** The app stores an asset ID. It never stores a script. An asset ID
> names something that is already computed under your own project. A script
> would be arbitrary compute on the Earth Engine budget of FRMS.

> **Caution** Name the band if the asset has more than one. Earth Engine refuses
> a palette on an image with several bands, and the layer will not render. Also
> set **Hide values below** when the no-data value of the asset is zero.
> Otherwise the whole world paints as the bottom of the gradient.

## Where to learn Earth Engine

Read [Learning](/guide/learning#earth-engine) for a short path into the Code
Editor, the data catalogue and the common mistakes.
