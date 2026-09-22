# Nexstrata Development Roadmap

## ✅ Completed Features (v1.0)

### Dashboard & User Experience
- [x] `V10-DASH-1` Customizable dashboard page for logged-in users
- [x] `V10-DASH-2` Drag-and-drop widget system (Saved Charts, Recent Jobs, Species List, Quick Filters, Environmental Stats)
- [x] `V10-DASH-3` Dashboard quick access link in user dropdown menu
- [x] `V10-DASH-4` Collapsible side panels (Active Layers & Layer Selection)
- [x] `V10-DASH-5` Global visibility toggle button
- [x] `V10-DASH-6` Advanced blend mode controls (hidden by default)
- [x] `V10-DASH-7` Simplified copy for non-technical audience (Home page & Guide)

### Data Import & Integration
- [x] `V10-DATA-1` Earth Engine asset import via asset path
- [x] `V10-DATA-2` GBIF export import tool (CSV → GeoJSON → GEE asset path)
- [x] `V10-DATA-3` Dataset validation and metadata extraction
- [x] `V10-DATA-4` Session-based dataset management

---

## 🚧 In Progress / Next Priority (v1.1)

### MaxEnt Modeling Suite - Core Infrastructure
- [x] `V11-CORE-1` **Modeling Core**: Implement `netlify/lib/maxent.mjs` with predictor registry, spatial cross-validation, and `amnhMaxent` integration
- [x] `V11-CORE-2` **Database Schema**: Create `model_configs`, `model_runs`, and `model_results` tables in Supabase
- [x] `V11-CORE-3` **Server Routes**:
  - `POST /api/modeling/maxent/train` - Submit training job to Earth Engine
  - `GET /api/modeling/maxent/results/:jobId` - Fetch training results and metrics
  - `GET /api/modeling/maxent/models` - List user's saved models
  - `DELETE /api/modeling/maxent/models/:id` - Delete a model
- [x] `V11-CORE-4` **Composable**: `useMaxEnt.ts` for state management and API communication
- [x] `V11-CORE-5` **Privacy Controls**: Implement data visibility settings (Private/Members-only/Public) using shared project access logic

### MaxEnt Modeling Suite - User Interface
- [x] `V11-UI-1` **Training Page** (`/pages/modeling/maxent.vue`):
  - Dataset selection with preview (record count, date range, bounds)
  - Predictor layer selection with correlation matrix warnings
  - Model parameter configuration (regularization, feature types, background points, CV folds)
  - Real-time job progress tracking
- [x] `V11-UI-2` **Results Visualization Components**:
  - `ResponseCurve.vue` - Plot species response to environmental variables
  - `VariableContribution.vue` - Bar chart of predictor importance
  - `ROCCurve.vue` - Model performance with AUC score
  - `ConfusionMatrix.vue` - Heatmap of prediction accuracy
- [x] `V11-UI-3` **Model Comparison Tool**: Side-by-side comparison of multiple model runs
- [x] `V11-UI-4` **Suitability Map Renderer**: Display MaxEnt output as interactive heatmap layer

### MaxEnt Heatmap Integration
- [ ] `V11-HEAT-1` Extend `HeatmapControls.vue` with "MaxEnt Suitability" type
- [ ] `V11-HEAT-2` Add probability/binary visualization toggle
- [ ] `V11-HEAT-3` Implement threshold slider for binary classification
- [ ] `V11-HEAT-4` Confidence interval overlay option
- [ ] `V11-HEAT-5` Layer Manager integration for MaxEnt outputs

---

## 📅 Future Enhancements (v1.2+)

### UI & UX Improvements
- [ ] `V12-UI-1` **Intuitive Navigation**: Streamline the path from data import to model training to reduce friction
- [ ] `V12-UI-2` **Contextual Onboarding**: Implement "empty state" guides and tooltips for complex modeling parameters
- [ ] `V12-UI-3` **Visual Hierarchy Refinement**: Improve contrast and layout of side panels for better focus on the map
- [ ] `V12-UI-4` **Interactive Data Previews**: Enhance dataset selection with instant visual summaries before committing to a model run

### Advanced Modeling Features
- [ ] `V12-MOD-1` **Ensemble Modeling**: Average predictions from multiple model runs
- [ ] `V12-MOD-2` **Projection Tools**: Project models to future climate scenarios (CMIP6 integration)
- [ ] `V12-MOD-3` **Batch Processing**: Train models for multiple species simultaneously
- [ ] `V12-MOD-4` **Model Export**: Download suitability rasters as GeoTIFF
- [ ] `V12-MOD-5` **Threshold Optimization**: Automatic threshold selection (MaxSSS, 10th percentile)

### Data Quality & Ethics
- [ ] `V12-ETH-1` **Sampling Bias Correction**: Automated thinning and bias file generation
- [ ] `V12-ETH-2` **Spatial Autocorrelation Checks**: Warn about clustered occurrence records
- [ ] `V12-ETH-3` **Extrapolation Risk Maps**: Highlight areas outside training environmental space (MOP/MEX analysis)
- [ ] `V12-ETH-4` **Sensitive Species Protection**: Automatic coordinate obscuring for threatened species

### Collaboration & Sharing
- [ ] `V12-COLL-1` **Public Model Gallery**: Browse and reuse models from other users
- [ ] `V12-COLL-2` **Team Workspaces**: Shared projects for research groups
- [ ] `V12-COLL-3` **Model Citation Generator**: Auto-generate citations for published models
- [ ] `V12-COLL-4` **Export to R/Python**: Generate reproducible scripts for external analysis

### Performance & Scalability
- [ ] `V12-PERF-1` **Job Queue System**: Manage long-running training jobs with email notifications
- [ ] `V12-PERF-2` **Result Caching**: Store frequently accessed model outputs
- [ ] `V12-PERF-3` **Lazy Loading**: Defer heavy chart components until needed
- [ ] `V12-PERF-4` **Mobile Optimization**: Responsive design for modeling interface on tablets

### Documentation & Onboarding
- [ ] `V12-DOC-1` **Interactive Tutorial**: Step-by-step walkthrough for first MaxEnt run
- [ ] `V12-DOC-2` **Video Guides**: Short screencasts for key workflows
- [ ] `V12-DOC-3` **Glossary Tooltips**: Hover explanations for technical terms (AUC, regularization, etc.)
- [ ] `V12-DOC-4` **Example Datasets**: Pre-loaded sample data for practice runs

---

## Technical Debt & Refactoring

- [x] `DEBT-1` **TypeScript Migration**: Convert remaining `.js` files to `.ts` for better type safety (Core composables migration substantially complete)
- [ ] `DEBT-2` **Test Coverage**:
  - Unit tests for all new MaxEnt components
  - Integration tests for full modeling pipeline
  - E2E tests for dashboard customization
- [ ] `DEBT-3` **Error Handling**: Standardize error messages and recovery flows
- [ ] `DEBT-4` **Accessibility Audit**: Ensure WCAG 2.1 compliance across new features
- [ ] `DEBT-5` **Performance Monitoring**: Add logging for Earth Engine job durations and failures

---

## Known Issues

- [ ] `ISSUE-1` LayerManager state persistence occasionally fails on mobile Safari
- [ ] `ISSUE-2` Large GBIF exports (>10k records) may timeout during import
- [ ] `ISSUE-3` Chart rendering slows with >50 data points in Saved Charts widget
- [ ] `ISSUE-4` Earth Engine asset validation doesn't check geometry types comprehensively

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
