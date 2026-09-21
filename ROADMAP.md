# Nexstrata Development Roadmap

## ✅ Completed Features (v1.0)

### Dashboard & User Experience
- [x] Customizable dashboard page for logged-in users
- [x] Drag-and-drop widget system (Saved Charts, Recent Jobs, Species List, Quick Filters, Environmental Stats)
- [x] Dashboard quick access link in user dropdown menu
- [x] Collapsible side panels (Active Layers & Layer Selection)
- [x] Global visibility toggle button
- [x] Advanced blend mode controls (hidden by default)
- [x] Simplified copy for non-technical audience (Home page & Guide)

### Data Import & Integration
- [x] Earth Engine asset import via asset path
- [x] GBIF export import tool (CSV → GeoJSON → GEE asset path)
- [x] Dataset validation and metadata extraction
- [x] Session-based dataset management

---

## 🚧 In Progress / Next Priority (v1.1)

### MaxEnt Modeling Suite - Core Infrastructure
- [ ] **Database Schema**: Create `model_configs`, `model_results`, and `model_runs` tables in Supabase
- [ ] **Server Routes**:
  - `POST /api/modeling/maxent/train` - Submit training job to Earth Engine
  - `GET /api/modeling/maxent/results/:jobId` - Fetch training results and metrics
  - `GET /api/modeling/maxent/models` - List user's saved models
  - `DELETE /api/modeling/maxent/models/:id` - Delete a model
- [ ] **Composable**: `useMaxEnt.js` for state management and API communication
- [ ] **Privacy Controls**: Implement data visibility settings (Private/Members-only/Public) with consent dialogs

### MaxEnt Modeling Suite - User Interface
- [ ] **Training Page** (`/pages/modeling/maxent.vue`):
  - Dataset selection with preview (record count, date range, bounds)
  - Predictor layer selection with correlation matrix warnings
  - Model parameter configuration (regularization, feature types, background points, CV folds)
  - Real-time job progress tracking
- [ ] **Results Visualization Components**:
  - `ResponseCurve.vue` - Plot species response to environmental variables
  - `VariableContribution.vue` - Bar chart of predictor importance
  - `ROCCurve.vue` - Model performance with AUC score
  - `ConfusionMatrix.vue` - Heatmap of prediction accuracy
- [ ] **Model Comparison Tool**: Side-by-side comparison of multiple model runs
- [ ] **Suitability Map Renderer**: Display MaxEnt output as interactive heatmap layer

### MaxEnt Heatmap Integration
- [ ] Extend `HeatmapControls.vue` with "MaxEnt Suitability" type
- [ ] Add probability/binary visualization toggle
- [ ] Implement threshold slider for binary classification
- [ ] Confidence interval overlay option
- [ ] Layer Manager integration for MaxEnt outputs

---

## 📅 Future Enhancements (v1.2+)

### Advanced Modeling Features
- [ ] **Ensemble Modeling**: Average predictions from multiple model runs
- [ ] **Projection Tools**: Project models to future climate scenarios (CMIP6 integration)
- [ ] **Batch Processing**: Train models for multiple species simultaneously
- [ ] **Model Export**: Download suitability rasters as GeoTIFF
- [ ] **Threshold Optimization**: Automatic threshold selection (MaxSSS, 10th percentile)

### Data Quality & Ethics
- [ ] **Sampling Bias Correction**: Automated thinning and bias file generation
- [ ] **Spatial Autocorrelation Checks**: Warn about clustered occurrence records
- [ ] **Extrapolation Risk Maps**: Highlight areas outside training environmental space (MOP/MEX analysis)
- [ ] **Sensitive Species Protection**: Automatic coordinate obscuring for threatened species

### Collaboration & Sharing
- [ ] **Public Model Gallery**: Browse and reuse models from other users
- [ ] **Team Workspaces**: Shared projects for research groups
- [ ] **Model Citation Generator**: Auto-generate citations for published models
- [ ] **Export to R/Python**: Generate reproducible scripts for external analysis

### Performance & Scalability
- [ ] **Job Queue System**: Manage long-running training jobs with email notifications
- [ ] **Result Caching**: Store frequently accessed model outputs
- [ ] **Lazy Loading**: Defer heavy chart components until needed
- [ ] **Mobile Optimization**: Responsive design for modeling interface on tablets

### Documentation & Onboarding
- [ ] **Interactive Tutorial**: Step-by-step walkthrough for first MaxEnt run
- [ ] **Video Guides**: Short screencasts for key workflows
- [ ] **Glossary Tooltips**: Hover explanations for technical terms (AUC, regularization, etc.)
- [ ] **Example Datasets**: Pre-loaded sample data for practice runs

---

## Technical Debt & Refactoring

- [ ] **TypeScript Migration**: Convert remaining `.js` files to `.ts` for better type safety
- [ ] **Test Coverage**:
  - Unit tests for all new MaxEnt components
  - Integration tests for full modeling pipeline
  - E2E tests for dashboard customization
- [ ] **Error Handling**: Standardize error messages and recovery flows
- [ ] **Accessibility Audit**: Ensure WCAG 2.1 compliance across new features
- [ ] **Performance Monitoring**: Add logging for Earth Engine job durations and failures

---

## Known Issues

- [ ] LayerManager state persistence occasionally fails on mobile Safari
- [ ] Large GBIF exports (>10k records) may timeout during import
- [ ] Chart rendering slows with >50 data points in Saved Charts widget
- [ ] Earth Engine asset validation doesn't check geometry types comprehensively

---

## Contribution Guidelines

1. **Branch Naming**: `feature/<name>`, `fix/<name>`, or `roadmap/<phase>`
2. **Commit Messages**: Follow conventional commits (`feat:`, `fix:`, `docs:`, etc.)
3. **Testing**: All new features require tests before merge
4. **Documentation**: Update guide and tooltips for user-facing changes
5. **Code Review**: At least one approval required for PRs to `master`

---

*Last Updated: December 2025*
*Current Version: v1.0 (Dashboard, UX Improvements, Data Import)*
*Next Milestone: v1.1 (MaxEnt Modeling Suite)*
