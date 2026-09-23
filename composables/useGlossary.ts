// Shared glossary definitions for technical terms used across the app.
// Each key maps to the canonical hover text shown in GlossaryTooltip.

export const GLOSSARY: Record<string, string> = {
  AUC: 'Area Under the ROC Curve. Measures how well the model separates presences from background. 0.5 = random; 1.0 = perfect. Values above 0.75 are generally useful.',
  ROC: 'Receiver Operating Characteristic curve. Plots the true positive rate against the false positive rate at every threshold. The further the curve bows toward the top-left, the better the model.',
  'AUC score': 'Area Under the ROC Curve — a single number summarising how well the model discriminates presences from background. 0.5 is random; 1.0 is perfect; above 0.75 is generally useful.',
  MaxEnt: 'Maximum Entropy modeling. Estimates the most uniform probability distribution consistent with the known presence locations and environmental constraints. Widely used for species distribution modeling from presence-only data.',
  'background points': 'Pseudo-absence locations sampled at random across the landscape. MaxEnt contrasts the environmental conditions at presence locations against these to learn which conditions the species favors.',
  regularization: 'A penalty added during model fitting that discourages overly complex responses. Higher values produce smoother, more general distributions; lower values allow tighter fits that may overfit to the training data.',
  'regularization multiplier': 'Controls model complexity. Values between 1 and 3 are typical. Increase if the suitability surface looks noisy or suspiciously follows the observation footprint.',
  NDVI: 'Normalized Difference Vegetation Index. A measure of vegetation greenness derived from near-infrared and red satellite bands. Ranges from −1 (no vegetation) to +1 (dense green cover).',
  'soil moisture': 'Long-run average top-layer soil wetness, expressed as volumetric water content (m³/m³). Derived from ERA5-Land reanalysis data.',
  'background count': 'The number of random pseudo-absence points drawn from across the landscape. More points produce a more stable contrast with the presences, but training takes longer. 1,000–10,000 is typical.',
  predictors: 'Environmental variables (elevation, rainfall, temperature, vegetation, etc.) fed into the model as inputs. More predictors is not always better — stick to variables with a plausible ecological link to your species.',
  'response curve': 'A plot showing how predicted suitability changes as one predictor varies while the others are held at their average. Helps interpret which direction of a variable the model finds favorable.',
  'variable contribution': 'The fraction of the model\'s total information gain attributable to each predictor. A high percentage means that variable was heavily used; low means it barely influenced the output.',
  'suitability surface': 'A raster where each pixel holds a predicted probability (0–1) that the environment at that location resembles the conditions where the species was observed.',
  enrichment: 'The process of sampling environmental data (elevation, climate, soil, vegetation) from Earth Engine at each observation\'s coordinates, adding those values as extra columns to the dataset.',
  'cross-validation': 'Model evaluation strategy that trains on a subset of data and tests on withheld points, repeated across multiple folds. Produces a less optimistic AUC than evaluating on the training data itself.',
  'habitat suitability': 'A dimensionless score (0–1) expressing how similar the local environment is to conditions where the species was recorded. It is not a probability of occurrence.',
}

export function useGlossary() {
  function define(term: string): string {
    return GLOSSARY[term] ?? GLOSSARY[term.toLowerCase()] ?? ''
  }
  return { define, GLOSSARY }
}
