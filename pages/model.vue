<template>
  <div class="model-page">
    <!-- ── Page tabs ─────────────────────────────────────────────────────── -->
    <nav class="page-tabs">
      <button class="ptab" :class="{ active: tab === 'new' }" @click="tab = 'new'">New Model</button>
      <button class="ptab" :class="{ active: tab === 'models' }" @click="tab = 'models'">Models</button>
    </nav>

    <!-- ════════════════════════════════════════════════════════════════════
         TAB: NEW MODEL (stepper)
    ═══════════════════════════════════════════════════════════════════════ -->
    <template v-if="tab === 'new'">
      <div class="stepper-header">
        <div class="stepper-title-row">
          <div>
            <h2>Model Pipeline</h2>
            <p class="sub">Import observations → enrich with environment data<template v-if="advancedMode"> → analyse predictors</template> → train a habitat model → evaluate results.</p>
          </div>
          <button class="mode-toggle" @click="advancedMode = !advancedMode" type="button">
            {{ advancedMode ? 'Advanced mode' : 'Quick mode' }}
          </button>
        </div>
      </div>

      <!-- Step indicator -->
      <nav class="steps">
        <button v-for="(s, i) in visibleSteps" :key="s.key" class="step-btn"
                :class="{ active: step === s.key, done: completedSteps.has(s.key), reachable: canReach(s.key) }"
                :disabled="!canReach(s.key)"
                @click="goStep(s.key)">
          <span class="step-num">{{ i + 1 }}</span>
          <span class="step-label">{{ s.label }}</span>
        </button>
      </nav>

      <ClientOnly>
        <!-- Auth gate -->
        <div v-if="membership.configured && !membership.isAuthed.value" class="gate">
          <p>Sign in to use the model pipeline.</p>
          <NuxtLink to="/login" class="btn primary">Sign in</NuxtLink>
        </div>
        <div v-else-if="membership.configured && !membership.isMember.value" class="gate">
          <p>The model pipeline is a membership benefit.</p>
          <button class="btn" :disabled="rechecking" @click="recheck">{{ rechecking ? 'Checking…' : 'Just joined? Check again' }}</button>
        </div>

        <template v-else>
          <!-- ═══════════ STEP 1 — SOURCE ═══════════════════════════════════ -->
          <section v-if="step === 'source'" class="panel">
            <h3>Observations</h3>
            <p class="hint">Choose where your species observations come from.</p>

            <div class="row">
              <label>Source type</label>
              <div class="sources">
                <label class="pick">
                  <input v-model="sourceForm.type" type="radio" value="dataset" />
                  <span>Use a saved dataset</span>
                </label>
                <label class="pick">
                  <input v-model="sourceForm.type" type="radio" value="bbox" />
                  <span>Filter by area, date &amp; taxon</span>
                </label>
                <label class="pick">
                  <input v-model="sourceForm.type" type="radio" value="fetch" />
                  <span>Import from iNaturalist / GBIF</span>
                </label>
              </div>
            </div>

            <!-- Saved dataset -->
            <template v-if="sourceForm.type === 'dataset'">
              <div class="row">
                <label for="src-dataset">Dataset</label>
                <select id="src-dataset" v-model="sourceForm.datasetSlug">
                  <option value="" disabled>Choose one…</option>
                  <option v-for="d in datasetsApi.datasets.value" :key="d.id" :value="d.slug">
                    {{ d.title }}
                  </option>
                </select>
              </div>
              <p v-if="!datasetsApi.datasets.value.length" class="hint warn">
                No saved datasets yet — try "Filter by area" or go to <NuxtLink to="/data">Data</NuxtLink> to import observations first.
              </p>
            </template>

            <!-- BBox with map-view shortcut -->
            <template v-else-if="sourceForm.type === 'bbox'">
              <div class="field-stack">
                <div class="field-row">
                  <span class="field-label">Region</span>
                  <div class="bbox-group">
                    <button class="btn secondary small" type="button" @click="useMapView">Use current map view</button>
                    <div class="bbox">
                      <label class="mini">N <input v-model.number="sourceForm.north" type="number" step="0.1" /></label>
                      <label class="mini">S <input v-model.number="sourceForm.south" type="number" step="0.1" /></label>
                      <label class="mini">W <input v-model.number="sourceForm.west" type="number" step="0.1" /></label>
                      <label class="mini">E <input v-model.number="sourceForm.east" type="number" step="0.1" /></label>
                    </div>
                  </div>
                </div>
                <div class="field-row">
                  <span class="field-label">Date filter</span>
                  <label class="pick pick-inline">
                    <input v-model="sourceForm.noDateFilter" type="checkbox" />
                    <span>No date filter (all time)</span>
                  </label>
                </div>
                <div v-if="!sourceForm.noDateFilter" class="field-row">
                  <span class="field-label"></span>
                  <div class="date-pair">
                    <label class="stack"><span>From</span><input v-model="sourceForm.dateFrom" type="date" /></label>
                    <label class="stack"><span>To</span><input v-model="sourceForm.dateTo" type="date" /></label>
                  </div>
                </div>
                <div class="field-row">
                  <span class="field-label">Taxon</span>
                  <TaxonAutocomplete id="src-taxon" v-model="sourceForm.taxon" placeholder="e.g. Amanita (optional)" source="inat" />
                </div>
              </div>
            </template>

            <!-- Fetch / import -->
            <template v-else-if="sourceForm.type === 'fetch'">
              <div class="field-stack">
                <div class="field-row">
                  <span class="field-label">Taxon</span>
                  <TaxonAutocomplete id="fetch-taxon" v-model="fetchForm.taxon"
                                     placeholder="e.g. Morchella, Amanitaceae" :disabled="fetchForm.loading"
                                     :source="fetchForm.source" />
                </div>
                <div class="field-row">
                  <span class="field-label">Source</span>
                  <div class="src-tabs">
                    <button v-for="s in FETCH_SOURCES" :key="s.key" class="src-tab"
                            :class="{ on: fetchForm.source === s.key }"
                            @click="fetchForm.source = s.key" type="button">{{ s.label }}</button>
                  </div>
                </div>
                <div class="field-row">
                  <span class="field-label">Date filter</span>
                  <label class="pick pick-inline">
                    <input v-model="fetchForm.noDateFilter" type="checkbox" :disabled="fetchForm.loading" />
                    <span>No date filter (all time)</span>
                  </label>
                </div>
                <div v-if="!fetchForm.noDateFilter" class="field-row">
                  <span class="field-label"></span>
                  <div class="date-pair">
                    <label class="stack"><span>From</span><input v-model="fetchForm.dateFrom" type="date" :disabled="fetchForm.loading" /></label>
                    <label class="stack"><span>To</span><input v-model="fetchForm.dateTo" type="date" :disabled="fetchForm.loading" /></label>
                  </div>
                </div>
              </div>
              <div class="actions" style="margin-top:0">
                <button class="btn secondary" :disabled="!fetchForm.taxon.trim() || fetchForm.loading"
                        @click="runFetch" type="button">
                  {{ fetchForm.loading ? 'Fetching…' : 'Fetch observations' }}
                </button>
              </div>
              <p v-if="fetchForm.error" class="msg error">{{ fetchForm.error }}</p>
              <p v-if="fetchForm.result" class="msg ok">
                {{ fetchForm.result.count }} records fetched as "{{ fetchForm.result.title }}" — ready to continue.
              </p>
            </template>

            <div class="actions">
              <button class="btn primary" :disabled="!canAdvanceSource" @click="advanceSource">
                Continue →
              </button>
            </div>
          </section>

          <!-- ═══════════ STEP 2 — ENRICH ══════════════════════════════════ -->
          <section v-else-if="step === 'enrich'" class="panel">
            <h3>Enrich with environment data</h3>
            <p class="hint">
              Sample terrain, climate, and vegetation values at each observation point.
              These become the predictors for the habitat model.
            </p>

            <div class="row">
              <label>Job name</label>
              <input v-model="enrichForm.title" type="text" placeholder="Autumn foray enrichment" />
            </div>

            <div class="row">
              <label>Environment layers</label>
              <div class="stages">
                <label v-for="s in stageList" :key="s.key" class="stage">
                  <input type="checkbox" :value="s.key" v-model="enrichForm.stages" />
                  <span><strong>{{ s.label }}</strong><em>{{ s.description }}</em></span>
                </label>
              </div>
            </div>

            <p v-if="enrichError" class="msg error">{{ enrichError }}</p>
            <p v-if="enrichNote" class="msg ok">{{ enrichNote }}</p>

            <div class="actions">
              <button class="btn secondary" @click="goStep('source')">← Back</button>
              <template v-if="!enrichJobId">
                <button class="btn primary"
                        :disabled="!enrichForm.stages.length || jobsApi.submitting.value"
                        @click="submitEnrich">
                  {{ jobsApi.submitting.value ? 'Queuing…' : 'Start enrichment' }}
                </button>
              </template>
              <template v-else>
                <span class="job-status" :class="enrichJobStatus">{{ enrichStatusLabel }}</span>
                <button v-if="enrichJobStatus === 'succeeded'" class="btn primary" @click="goStep(advancedMode ? 'explore' : 'train')">
                  {{ advancedMode ? 'Analyse predictors →' : 'Configure training →' }}
                </button>
              </template>
            </div>

            <div v-if="enrichJob" class="job-progress">
              <div class="prog-bar"><div class="prog-fill" :style="{ width: (enrichJob.progress || 0) + '%' }"></div></div>
              <p class="prog-msg">{{ enrichJob.message || enrichJob.stage || enrichStatusLabel }}</p>
            </div>
          </section>

          <!-- ═══════════ STEP 3 — EXPLORE (advanced mode only) ════════════ -->
          <section v-else-if="step === 'explore'" class="panel">
            <h3>Analyse predictors</h3>
            <p class="hint">
              Review which environment variables have good coverage and are not highly correlated
              before committing to a full training run.
            </p>

            <div v-if="!exploreData" class="explore-section">
              <h4>Load a dataset</h4>
              <div class="field-row" style="gap:0.5rem;align-items:center">
                <select v-model="exploreSlug" style="flex:1">
                  <option value="" disabled>Choose dataset…</option>
                  <option v-for="d in datasetsApi.datasets.value" :key="d.id" :value="d.slug">{{ d.title }}</option>
                </select>
                <button class="btn primary" :disabled="!exploreSlug || exploreLoading" @click="loadExploreData(exploreSlug)">
                  {{ exploreLoading ? 'Loading…' : 'Load' }}
                </button>
              </div>
              <p v-if="exploreError" class="msg error">{{ exploreError }}</p>
            </div>

            <template v-if="exploreData">
              <div class="explore-section">
                <h4>Variable coverage</h4>
                <table class="cov-table">
                  <thead><tr><th>Variable</th><th>Coverage</th><th>Recommend</th></tr></thead>
                  <tbody>
                    <tr v-for="v in exploreData.coverage" :key="v.key" :class="{ low: v.pct < 60 }">
                      <td>{{ v.label }}</td>
                      <td>
                        <span class="cov-bar-wrap">
                          <span class="cov-bar" :style="{ width: v.pct + '%', background: v.pct >= 80 ? '#22c55e' : v.pct >= 60 ? '#f59e0b' : '#ef4444' }"></span>
                          <span class="cov-pct">{{ v.pct.toFixed(0) }}%</span>
                        </span>
                      </td>
                      <td>
                        <span v-if="v.pct >= 80" class="badge ok">✓ Use</span>
                        <span v-else-if="v.pct >= 60" class="badge warn">Consider</span>
                        <span v-else class="badge bad">Skip</span>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <div class="explore-section">
                <h4>Variable importance <span class="badge-sub">(MaxEnt scout)</span></h4>
                <div v-if="!scoutContributions.length && !scoutRunning" class="actions" style="padding:0;margin-bottom:0.5rem">
                  <button class="btn secondary" @click="runScout">Run scout analysis</button>
                </div>
                <p v-if="scoutRunning" class="msg">Scout running… {{ scoutProgress ? scoutProgress + '%' : '' }}</p>
                <p v-if="scoutError" class="msg error">{{ scoutError }}</p>
                <div v-if="scoutContributions.length" class="contrib-list">
                  <div v-for="c in scoutContributions" :key="c.variable" class="contrib-row">
                    <span class="contrib-label">{{ c.variable }}</span>
                    <span class="contrib-bar-wrap">
                      <span class="contrib-bar" :style="{ width: c.pct + '%' }"></span>
                    </span>
                    <span class="contrib-pct">{{ c.pct.toFixed(1) }}%</span>
                  </div>
                </div>
              </div>

              <div v-if="exploreData.correlations.length && !scoutContributions.length" class="explore-section">
                <h4>Correlation between predictors</h4>
                <table class="corr-table">
                  <thead>
                    <tr>
                      <th></th>
                      <th v-for="v in exploreData.coverage" :key="v.key">{{ v.shortLabel }}</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="(row, ri) in exploreData.correlations" :key="ri">
                      <td class="row-label">{{ exploreData.coverage[ri]?.shortLabel }}</td>
                      <td v-for="(val, ci) in row" :key="ci"
                          class="corr-cell"
                          :class="{ high: Math.abs(val) > 0.7 && ri !== ci }"
                          :style="{ background: corrColor(val, ri === ci) }">
                        {{ ri === ci ? '—' : val.toFixed(2) }}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <div class="explore-section">
                <h4>Recommended predictors</h4>
                <div class="stages">
                  <label v-for="p in predictorList" :key="p.key" class="stage">
                    <input type="checkbox" :value="p.key" v-model="trainForm.predictors" />
                    <span><strong>{{ p.label }}</strong></span>
                  </label>
                </div>
              </div>
            </template>

            <div class="actions">
              <button class="btn secondary" @click="goStep('enrich')">← Back</button>
              <button class="btn primary" :disabled="trainForm.predictors.length < 2" @click="goStep('train')">
                Configure training →
              </button>
            </div>
          </section>

          <!-- ═══════════ STEP 4 — TRAIN ════════════════════════════════════ -->
          <section v-else-if="step === 'train'" class="panel">
            <h3>Train habitat model</h3>
            <p class="hint">Configure and start the MaxEnt training run.</p>

            <div class="row">
              <label for="train-title">Model name</label>
              <input id="train-title" v-model="trainForm.title" type="text" placeholder="Front Range fungi – autumn 2024" />
            </div>

            <div class="row">
              <label>Predictors</label>
              <div class="stages">
                <label v-for="p in predictorList" :key="p.key" class="stage">
                  <input type="checkbox" :value="p.key" v-model="trainForm.predictors" />
                  <span><strong>{{ p.label }}</strong></span>
                </label>
              </div>
            </div>

            <div class="row">
              <label for="bg-count">Background points</label>
              <input id="bg-count" v-model.number="trainForm.background" type="number" min="100" max="10000" step="100" />
              <p class="hint">Random absence proxies. 1 000–5 000 is typical.</p>
            </div>

            <div class="row">
              <label>Options</label>
              <label class="pick">
                <input type="checkbox" v-model="trainForm.autoOptimize" />
                <span>Auto-select best predictors (scout run)</span>
              </label>
              <label class="pick">
                <input type="checkbox" v-model="trainForm.preciseOnly" />
                <span>Precise GPS coordinates only</span>
              </label>
            </div>

            <div class="row">
              <label>Projection region <span class="opt-label">optional</span></label>
              <div class="bbox">
                <label class="mini">N <input v-model.number="trainForm.regionNorth" type="number" step="0.1" /></label>
                <label class="mini">S <input v-model.number="trainForm.regionSouth" type="number" step="0.1" /></label>
                <label class="mini">W <input v-model.number="trainForm.regionWest" type="number" step="0.1" /></label>
                <label class="mini">E <input v-model.number="trainForm.regionEast" type="number" step="0.1" /></label>
              </div>
            </div>

            <p v-if="trainError" class="msg error">{{ trainError }}</p>

            <div class="actions">
              <button class="btn secondary" @click="goStep(advancedMode ? 'explore' : 'enrich')">← Back</button>
              <template v-if="!trainJobId">
                <button class="btn primary"
                        :disabled="trainForm.predictors.length < 2 || !trainForm.title || maxEnt.pending.value"
                        @click="submitTrain">
                  {{ maxEnt.pending.value ? 'Queuing…' : 'Train model' }}
                </button>
              </template>
              <template v-else>
                <span class="job-status" :class="trainStatus">{{ trainStatusLabel }}</span>
                <button v-if="trainStatus === 'succeeded'" class="btn primary" @click="goStep('evaluate')">
                  Evaluate model →
                </button>
              </template>
            </div>

            <div v-if="maxEnt.activeJob.value" class="job-progress">
              <div class="prog-bar"><div class="prog-fill indeterminate"></div></div>
              <p class="prog-msg">{{ maxEnt.activeJob.value.status === 'pending' ? 'Queued…' : 'Training in progress — this takes a few minutes.' }}</p>
            </div>
          </section>

          <!-- ═══════════ STEP 5 — EVALUATE ═════════════════════════════════ -->
          <section v-else-if="step === 'evaluate'" class="panel">
            <h3>Evaluate model</h3>
            <p class="hint">
              Review how well the model performed before rendering the full suitability map.
              A good model has AUC &gt; 0.75.
            </p>

            <div v-if="!evalData && !evalLoading && !evalError" class="actions">
              <button class="btn primary" @click="loadEval">Load evaluation data</button>
            </div>

            <p v-if="evalLoading" class="msg">Running evaluation — this may take 30–60 seconds…</p>
            <p v-if="evalError" class="msg error">{{ evalError }}</p>

            <template v-if="evalData || trainedModel">
              <div class="eval-summary">
                <div class="eval-stat" :class="aucGrade">
                  <span class="stat-value">{{ aucDisplay }}</span>
                  <span class="stat-label">AUC</span>
                </div>
                <div class="eval-stat">
                  <span class="stat-value">{{ gradeDisplay }}</span>
                  <span class="stat-label">Grade</span>
                </div>
                <div v-if="evalData?.presenceCount" class="eval-stat">
                  <span class="stat-value">{{ evalData.presenceCount }}</span>
                  <span class="stat-label">Presences</span>
                </div>
              </div>

              <p class="hint" :class="aucGrade">
                <template v-if="aucGrade === 'excellent'">Excellent — this model distinguishes habitat from non-habitat very well.</template>
                <template v-else-if="aucGrade === 'good'">Good — model performance is solid.</template>
                <template v-else-if="aucGrade === 'fair'">Fair — the model may be usable but treat predictions cautiously.</template>
                <template v-else-if="aucGrade === 'weak'">Weak — consider adding more observations or adjusting predictors before rendering.</template>
              </p>

              <div v-if="contributions.length" class="explore-section">
                <h4>Predictor contributions</h4>
                <VariableContribution :data="contributions" title="" x-label="Contribution (%)" />
              </div>

              <div v-if="evalData?.roc" class="explore-section">
                <h4>ROC curve</h4>
                <ROCCurve :points="evalData.roc" :auc="evalData.auc" />
              </div>
            </template>

            <div class="actions">
              <button class="btn secondary" @click="goStep('train')">← Back</button>
              <button v-if="trainedModel" class="btn primary" @click="openOnMap">
                View on map →
              </button>
              <button class="btn secondary" @click="tab = 'models'">
                All models
              </button>
            </div>
          </section>
        </template>
      </ClientOnly>
    </template>

    <!-- ════════════════════════════════════════════════════════════════════
         TAB: MODELS (gallery)
    ═══════════════════════════════════════════════════════════════════════ -->
    <template v-else-if="tab === 'models'">
      <MaxEntTutorial ref="tutorialRef" />

      <div class="head">
        <div class="head-text">
          <h2>Habitat Suitability Models</h2>
          <p class="sub">Your trained models. <button class="linkish" @click="tab = 'new'">Train a new model →</button></p>
        </div>
        <div class="head-actions">
          <button class="btn ghost small" @click="tutorialRef?.start()">Tutorial</button>
          <button v-if="selectedModels.length > 0" class="btn primary small" @click="showComparison = true">
            Compare ({{ selectedModels.length }})
          </button>
        </div>
      </div>

      <ClientOnly>
        <div v-if="membership.configured && !membership.isMember.value" class="gate">
          <p>MaxEnt modeling is a membership benefit.</p>
          <NuxtLink to="/login" class="btn primary">Sign in</NuxtLink>
        </div>

        <template v-else>
          <div v-if="showComparison" class="overlay" @click.self="showComparison = false">
            <div class="overlay-content">
              <button class="close-btn" @click="showComparison = false">✕</button>
              <ModelComparison :selected="selectedModels" />
            </div>
          </div>

          <div v-if="maxEnt.activeJob.value" class="job-banner">
            <div class="banner-body">
              <div class="banner-info">
                <strong class="banner-name">{{ maxEnt.activeJob.value.model_configs?.title }}</strong>
                <span class="status-pill" :class="maxEnt.activeJob.value.status">{{ maxEnt.activeJob.value.status }}</span>
              </div>
              <div class="banner-bar">
                <div class="bar-fill" :style="{ width: bannerProgress + '%' }"></div>
              </div>
            </div>
          </div>

          <section class="panel">
            <div class="panel-head">
              <h3>Your Models</h3>
              <div class="head-acts">
                <button class="btn primary small" @click="tab = 'new'">+ New Model</button>
                <button class="btn-text" :disabled="modelsLoading" @click="maxEnt.fetchModels()">Refresh</button>
              </div>
            </div>

            <p v-if="modelsLoading" class="loading-msg">Loading…</p>

            <div v-else-if="!maxEnt.models.value.length" class="empty-state">
              <p class="empty-title">No models yet</p>
              <p class="empty-sub">Train a model to get started.</p>
              <button class="btn primary" @click="tab = 'new'">Train your first model</button>
            </div>

            <div v-else class="model-grid">
              <div v-for="m in maxEnt.models.value" :key="m.id" class="model-card" :class="{ selected: m.selected }">
                <div class="card-top">
                  <label class="card-check">
                    <input type="checkbox" v-model="m.selected" />
                    <span class="card-title">{{ m.title }}</span>
                  </label>
                  <span class="vis-pill" :class="m.visibility">{{ VISIBILITY_LABELS[m.visibility] }}</span>
                </div>
                <div class="card-meta">
                  <template v-if="m.predictors?.length">
                    <span>{{ m.predictors.length }} predictors</span>
                    <span class="dot">·</span>
                    <span>{{ (m.background_count || 0).toLocaleString() }} bg pts</span>
                  </template>
                  <span v-else class="reg-tag">Registered</span>
                  <template v-if="m.results?.[0]?.auc != null">
                    <span class="dot">·</span>
                    <span class="auc-val" :class="m.results[0].grade">{{ m.results[0].auc.toFixed(3) }} AUC</span>
                  </template>
                </div>
                <div class="card-footer">
                  <span class="card-date">{{ fmtWhen(m.created_at) }}</span>
                  <div class="card-actions">
                    <button class="btn small primary" @click="openModelOnMap(m)">View on Map</button>
                    <button class="btn small danger" @click="confirmDelete(m)">Delete</button>
                  </div>
                </div>
              </div>
            </div>
          </section>

          <!-- Link pre-computed asset (advanced) -->
          <details class="adv-panel">
            <summary class="adv-summary">Link a pre-computed Earth Engine asset</summary>
            <div class="adv-body">
              <p class="tab-desc">Add a suitability surface already computed in Earth Engine by asset path.</p>
              <div class="form-row">
                <label for="reg-title">Model name</label>
                <input id="reg-title" v-model="registerForm.title" type="text" placeholder="e.g. Red-belted Conk — 2024 run" />
              </div>
              <div class="form-row">
                <label for="reg-asset">Earth Engine asset path</label>
                <input id="reg-asset" v-model="registerForm.asset_path" type="text" placeholder="projects/…/assets/…" />
              </div>
              <div class="form-row">
                <label for="reg-desc">Description <em class="opt-label">optional</em></label>
                <textarea id="reg-desc" v-model="registerForm.description" placeholder="Notes about this run…" rows="2"></textarea>
              </div>
              <div class="form-row narrow">
                <label>Visibility</label>
                <select v-model="registerForm.visibility">
                  <option v-for="v in visibilities" :key="v" :value="v">{{ VISIBILITY_LABELS[v] }}</option>
                </select>
              </div>
              <div class="form-actions">
                <button class="btn primary"
                        :disabled="maxEnt.pending.value || !registerForm.title.trim() || !registerForm.asset_path.trim()"
                        @click="onRegister">
                  {{ maxEnt.pending.value ? 'Registering…' : 'Register asset' }}
                </button>
              </div>
              <p v-if="registerError" class="form-msg error">{{ registerError }}</p>
              <p v-if="registerNote" class="form-msg ok">{{ registerNote }}</p>
            </div>
          </details>
        </template>
      </ClientOnly>
    </template>
  </div>
</template>

<script setup>
import { useEeJobs } from '~/composables/useEeJobs'
import { useDatasets, VISIBILITY_LABELS } from '~/composables/useDatasets'
import { useMaxEnt } from '~/composables/useMaxEnt'
import MaxEntTutorial from '~/components/MaxEntTutorial.vue'
import ModelComparison from '~/components/ModelComparison.vue'

// ── Tab + mode state ──────────────────────────────────────────────────────────
const route = useRoute()
const router = useRouter()

const tab = computed({
  get: () => (route.query.tab === 'models' ? 'models' : 'new'),
  set: (v) => router.replace({ query: { ...route.query, tab: v === 'models' ? 'models' : undefined, step: undefined } }),
})

const advancedMode = ref(false)

// ── Steps ─────────────────────────────────────────────────────────────────────
const ALL_STEPS = [
  { key: 'source',   label: 'Source' },
  { key: 'enrich',   label: 'Enrich' },
  { key: 'explore',  label: 'Explore' },
  { key: 'train',    label: 'Train' },
  { key: 'evaluate', label: 'Evaluate' },
]

const visibleSteps = computed(() =>
  advancedMode.value ? ALL_STEPS : ALL_STEPS.filter(s => s.key !== 'explore')
)

const STEP_ORDER = computed(() => visibleSteps.value.map(s => s.key))

const step = computed({
  get: () => {
    const q = route.query.step
    return STEP_ORDER.value.includes(q) ? q : 'source'
  },
  set: (v) => router.replace({ query: { ...route.query, step: v } }),
})

const completedSteps = ref(new Set())

function canReach(key) {
  const idx = STEP_ORDER.value.indexOf(key)
  if (idx === 0) return true
  const prev = STEP_ORDER.value[idx - 1]
  if (completedSteps.value.has(prev)) return true
  if (key === 'explore') return !!(enrichDatasetSlug.value || sourceForm.datasetSlug)
  return false
}

function goStep(key) {
  if (!canReach(key)) return
  if (key === 'explore') {
    const slug = enrichDatasetSlug.value || sourceForm.datasetSlug || ''
    if (slug && !exploreData.value) { exploreSlug.value = slug; loadExploreData(slug) }
    else if (!exploreSlug.value) { exploreSlug.value = slug }
  }
  step.value = key
}

// ── Composables ───────────────────────────────────────────────────────────────
const membership = useMembership()
const jobsApi = useEeJobs()
const datasetsApi = useDatasets()
const maxEnt = useMaxEnt()
useFilters()

const tutorialRef = ref(null)
const showComparison = ref(false)
const modelsLoading = ref(false)
const selectedModels = computed(() => maxEnt.models.value.filter(m => m.selected))
const modelOverlay = useModelOverlay()

const rechecking = ref(false)
async function recheck() {
  rechecking.value = true
  try { await membership.refresh?.() } finally { rechecking.value = false }
}

onMounted(async () => {
  datasetsApi.refresh()
  modelsLoading.value = true
  try { await maxEnt.fetchModels() } finally { modelsLoading.value = false }
})

// ── Predictor / stage metadata ─────────────────────────────────────────────────
const stageList = [
  { key: 'terrain',       label: 'Terrain',                description: 'Elevation, slope, aspect and exposure indices.' },
  { key: 'landcover',     label: 'Land cover',             description: 'ESA WorldCover class at each point (10 m).' },
  { key: 'soil_moisture', label: 'Soil moisture',          description: 'ERA5-Land volumetric soil water on the day of the record.' },
  { key: 'precip',        label: 'Rainfall lead-up',       description: 'CHIRPS daily rainfall for the 7 days up to each record.' },
  { key: 'temperature',   label: 'Temperature lead-up',    description: 'ERA5-Land daily max/min for the 7 days up to each record.' },
  { key: 'ndvi',          label: 'Vegetation (NDVI)',      description: 'Sentinel-2 NDVI and moisture index, cloud-screened.' },
  { key: 'soil',          label: 'Soil properties',        description: 'USDA texture class, percent sand, depth to bedrock.' },
  { key: 'soil_taxonomy', label: 'Soil taxonomy',          description: 'USDA great group and order at each point.' },
  { key: 'fire',          label: 'Fire history',           description: 'Most recent burn year and years since fire (MODIS, 2001+).' },
  { key: 'forest',        label: 'Forest type & structure', description: 'GAP forest type, canopy cover and stand height (CONUS).' },
]

const predictorList = [
  { key: 'elevation',     label: 'Elevation' },
  { key: 'slope',         label: 'Slope' },
  { key: 'aspect',        label: 'Aspect' },
  { key: 'ndvi',          label: 'Vegetation (NDVI)' },
  { key: 'soil_moisture', label: 'Soil moisture' },
  { key: 'precip_normal', label: 'Precipitation' },
  { key: 'temp_normal',   label: 'Temperature' },
]

const PREDICTOR_META = {
  elevation:     { label: 'Elevation',     shortLabel: 'Elev' },
  slope:         { label: 'Slope',         shortLabel: 'Slope' },
  aspect:        { label: 'Aspect',        shortLabel: 'Asp' },
  ndvi:          { label: 'NDVI',          shortLabel: 'NDVI' },
  soil_moisture: { label: 'Soil moisture', shortLabel: 'Soil' },
  precip_normal: { label: 'Precipitation', shortLabel: 'Prec' },
  temp_normal:   { label: 'Temperature',   shortLabel: 'Temp' },
}

// ── Step 1: Source ─────────────────────────────────────────────────────────────
function defaultDateFrom() {
  const d = new Date(); d.setFullYear(d.getFullYear() - 2); return d.toISOString().slice(0, 10)
}
function defaultDateTo() { return new Date().toISOString().slice(0, 10) }

const sourceForm = reactive({
  type: 'dataset',
  datasetSlug: '',
  north: 41.0, south: 37.0, west: -109.1, east: -102.0,
  dateFrom: defaultDateFrom(), dateTo: defaultDateTo(),
  noDateFilter: false,
  taxon: '',
})

function useMapView() {
  try {
    const stored = JSON.parse(sessionStorage.getItem('mapView') || '{}')
    if (stored.north != null) {
      sourceForm.north = parseFloat(stored.north.toFixed(4))
      sourceForm.south = parseFloat(stored.south.toFixed(4))
      sourceForm.east  = parseFloat(stored.east.toFixed(4))
      sourceForm.west  = parseFloat(stored.west.toFixed(4))
    }
  } catch (_) {}
}

const canAdvanceSource = computed(() => {
  if (sourceForm.type === 'dataset') return !!sourceForm.datasetSlug
  if (sourceForm.type === 'bbox') return sourceForm.north != null && sourceForm.south != null && sourceForm.east != null && sourceForm.west != null
  if (sourceForm.type === 'fetch') return !!sourceForm.datasetSlug
  return false
})

const FETCH_SOURCES = [
  { key: 'auto', label: 'Auto' },
  { key: 'inat', label: 'iNaturalist' },
  { key: 'gbif', label: 'GBIF' },
]

const fetchForm = reactive({
  taxon: '', source: 'auto',
  dateFrom: defaultDateFrom(), dateTo: defaultDateTo(),
  noDateFilter: false, loading: false, error: '', result: null,
})

async function runFetch() {
  fetchForm.error = ''; fetchForm.result = null; fetchForm.loading = true
  try {
    const { accessToken } = useAuth()
    const token = await accessToken()
    const headers = token ? { authorization: `Bearer ${token}` } : {}
    const p = new URLSearchParams({ species: fetchForm.taxon.trim() })
    if (!fetchForm.noDateFilter && fetchForm.dateFrom) p.set('d1', fetchForm.dateFrom)
    if (!fetchForm.noDateFilter && fetchForm.dateTo) p.set('d2', fetchForm.dateTo)
    const src = fetchForm.source === 'auto' ? 'inat' : fetchForm.source
    const fn = src === 'gbif' ? 'gbif-fetch' : 'fetch-species'
    const res = await fetch(`/.netlify/functions/${fn}?${p.toString()}`, { headers })
    const data = await res.json().catch(() => ({}))
    if (!res.ok || !data.ok) throw new Error(data.error || `Fetch failed (${res.status})`)
    if (!data.count) throw new Error('No records found for that taxon.')
    const storagePath = `species/${data.slug}.geojson`
    const title = `${fetchForm.taxon.trim()} (${src === 'gbif' ? 'GBIF' : 'iNat'}, ${data.count})`
    const saveRes = await fetch('/.netlify/functions/datasets', {
      method: 'POST',
      headers: { 'content-type': 'application/json', ...headers },
      body: JSON.stringify({ action: 'save_fetched', path: storagePath, slug: data.slug, title, feature_count: data.count }),
    })
    const saved = await saveRes.json().catch(() => ({}))
    if (!saveRes.ok || !saved.ok) throw new Error(saved.error || 'Could not register the fetched dataset.')
    sourceForm.datasetSlug = saved.dataset.slug
    fetchForm.result = { count: data.count, title }
    await datasetsApi.refresh()
  } catch (e) {
    fetchForm.error = e?.message || 'Fetch failed.'
  } finally {
    fetchForm.loading = false
  }
}

function advanceSource() {
  if (!canAdvanceSource.value) return
  completedSteps.value = new Set([...completedSteps.value, 'source'])
  goStep('enrich')
}

// ── Step 2: Enrich ─────────────────────────────────────────────────────────────
const enrichForm = reactive({ title: '', stages: ['terrain', 'landcover', 'soil_moisture', 'precip', 'temperature', 'ndvi'] })
const enrichError = ref(''); const enrichNote = ref('')
const enrichJobId = ref(null); const enrichDatasetSlug = ref(null)

const enrichJob = computed(() => enrichJobId.value ? jobsApi.jobs.value.find(j => j.id === enrichJobId.value) : null)
const enrichJobStatus = computed(() => enrichJob.value?.status || (enrichJobId.value ? 'pending' : null))
const enrichStatusLabel = computed(() => {
  const s = enrichJobStatus.value
  if (s === 'succeeded') return 'Enrichment complete ✓'
  if (s === 'failed') return 'Enrichment failed'
  if (s === 'running') return `Running (${enrichJob.value?.progress || 0}%)…`
  return 'Queued…'
})

watch(enrichJobStatus, async (s) => {
  if (s === 'succeeded') {
    completedSteps.value = new Set([...completedSteps.value, 'enrich'])
    loadExploreData()
    try {
      const saved = await datasetsApi.saveJob(
        { id: enrichJobId.value, title: enrichForm.title || 'Pipeline enrichment' },
        { title: enrichForm.title || 'Pipeline enrichment' },
      )
      enrichDatasetSlug.value = saved.slug
    } catch (_) {}
  }
})

function buildSource() {
  if (sourceForm.type === 'dataset') return { type: 'dataset', slug: sourceForm.datasetSlug }
  return {
    type: 'bbox',
    bounds: { north: sourceForm.north, south: sourceForm.south, east: sourceForm.east, west: sourceForm.west },
    dateFrom: sourceForm.noDateFilter ? undefined : (sourceForm.dateFrom || undefined),
    dateTo: sourceForm.noDateFilter ? undefined : (sourceForm.dateTo || undefined),
    taxon: sourceForm.taxon || undefined,
  }
}

async function submitEnrich() {
  enrichError.value = ''; enrichNote.value = ''
  const spec = { kind: 'enrich', title: enrichForm.title || 'Pipeline enrichment', stages: enrichForm.stages, source: buildSource() }
  try {
    const result = await jobsApi.submit(spec)
    enrichJobId.value = result.job?.id || result.id
    enrichNote.value = 'Job queued — enrichment will run on the server.'
    pollEnrichJob()
  } catch (e) {
    enrichError.value = e?.message || 'Submission failed.'
  }
}

let enrichPollTimer = null
function pollEnrichJob() {
  if (enrichPollTimer) clearInterval(enrichPollTimer)
  enrichPollTimer = setInterval(async () => {
    await jobsApi.refresh()
    const job = jobsApi.jobs.value.find(j => j.id === enrichJobId.value)
    if (job && ['succeeded', 'failed', 'cancelled'].includes(job.status)) {
      clearInterval(enrichPollTimer); enrichPollTimer = null
    }
  }, 4000)
}
onUnmounted(() => { if (enrichPollTimer) clearInterval(enrichPollTimer) })

// ── Step 3: Explore ────────────────────────────────────────────────────────────
const exploreData = ref(null); const exploreSlug = ref(''); const exploreLoading = ref(false); const exploreError = ref('')
const scoutRunning = ref(false); const scoutProgress = ref(0); const scoutContributions = ref([]); const scoutError = ref('')
let scoutPollTimer = null

async function loadExploreData(slugOverride) {
  const slug = slugOverride || enrichDatasetSlug.value || (sourceForm.type === 'dataset' ? sourceForm.datasetSlug : null)
  if (!slug) return
  exploreLoading.value = true; exploreError.value = ''
  try {
    const { accessToken } = useAuth()
    const token = await accessToken()
    const headers = token ? { authorization: `Bearer ${token}` } : {}
    const res = await fetch(`/.netlify/functions/datasets?slug=${encodeURIComponent(slug)}`, { headers })
    if (!res.ok) { exploreError.value = `Failed to load dataset (${res.status})`; return }
    const data = await res.json()
    const features = data.geojson?.features || data.features || []
    if (!features.length) { exploreError.value = 'Dataset is empty.'; return }
    exploreData.value = computeExploreStats(features)
    scoutContributions.value = []
    const recommended = exploreData.value.coverage.filter(v => v.pct >= 80).map(v => v.key)
    trainForm.predictors = recommended.length >= 2 ? recommended : predictorList.map(p => p.key)
  } catch (e) {
    exploreError.value = e?.message || 'Failed to load dataset.'
  } finally {
    exploreLoading.value = false
  }
}

async function runScout() {
  if (scoutRunning.value) return
  scoutRunning.value = true; scoutError.value = ''; scoutProgress.value = 0; scoutContributions.value = []
  if (scoutPollTimer) { clearInterval(scoutPollTimer); scoutPollTimer = null }
  const spec = {
    kind: 'model', title: 'Scout pre-train',
    predictors: trainForm.predictors.length >= 2 ? trainForm.predictors : predictorList.map(p => p.key),
    background: 200, autoOptimize: false,
    source: enrichDatasetSlug.value ? { type: 'dataset', slug: enrichDatasetSlug.value } : buildSource(),
  }
  try {
    const result = await maxEnt.trainModel(spec)
    if (!result.ok) { scoutError.value = result.error || 'Scout submission failed.'; scoutRunning.value = false; return }
    scoutPollTimer = setInterval(async () => {
      const job = maxEnt.activeJob.value
      if (!job) return
      scoutProgress.value = job.progress || 0
      if (job.status === 'succeeded') {
        clearInterval(scoutPollTimer); scoutPollTimer = null; scoutRunning.value = false
        const contribs = job.eval_data?.contributions || job.contributions || []
        if (contribs.length) {
          const total = contribs.reduce((s, c) => s + (c.contribution ?? c.pct ?? 0), 0) || 1
          scoutContributions.value = contribs
            .map(c => ({ variable: c.variable || c.name || c.predictor || '', pct: ((c.contribution ?? c.pct ?? 0) / total) * 100 }))
            .sort((a, b) => b.pct - a.pct)
        } else {
          scoutError.value = 'Scout completed but returned no contributions.'
        }
      } else if (job.status === 'failed') {
        clearInterval(scoutPollTimer); scoutPollTimer = null; scoutRunning.value = false
        scoutError.value = job.error_message || 'Scout run failed.'
      }
    }, 5000)
  } catch (e) {
    scoutError.value = e?.message || 'Scout run failed.'; scoutRunning.value = false
  }
}
onUnmounted(() => { if (scoutPollTimer) clearInterval(scoutPollTimer) })

function computeExploreStats(features) {
  const total = features.length
  const keys = Object.keys(PREDICTOR_META)
  const coverage = keys.map(key => {
    const present = features.filter(f => f.properties?.[key] != null && !isNaN(f.properties[key])).length
    return { key, label: PREDICTOR_META[key].label, shortLabel: PREDICTOR_META[key].shortLabel, pct: total ? (present / total) * 100 : 0 }
  }).filter(v => v.pct > 0)
  const correlations = computeCorrelationMatrix(features, coverage.map(v => v.key))
  return { total, coverage, correlations }
}

function computeCorrelationMatrix(features, keys) {
  const cols = keys.map(k => features.map(f => f.properties?.[k] ?? null).filter(v => v != null))
  const n = keys.length
  const matrix = Array.from({ length: n }, () => new Array(n).fill(0))
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      if (i === j) { matrix[i][j] = 1; continue }
      const xi = cols[i], xj = cols[j], len = Math.min(xi.length, xj.length)
      if (len < 3) { matrix[i][j] = matrix[j][i] = 0; continue }
      const mx = xi.slice(0, len).reduce((a, b) => a + b, 0) / len
      const my = xj.slice(0, len).reduce((a, b) => a + b, 0) / len
      let num = 0, dx = 0, dy = 0
      for (let k = 0; k < len; k++) { const a = xi[k] - mx, b = xj[k] - my; num += a * b; dx += a * a; dy += b * b }
      matrix[i][j] = matrix[j][i] = (dx && dy) ? parseFloat((num / Math.sqrt(dx * dy)).toFixed(3)) : 0
    }
  }
  return matrix
}

function corrColor(val, isDiag) {
  if (isDiag) return 'transparent'
  const a = Math.abs(val)
  if (a > 0.7) return `rgba(239,68,68,${0.3 + a * 0.4})`
  if (a > 0.4) return `rgba(251,191,36,${0.2 + a * 0.3})`
  return `rgba(34,197,94,${0.1 + a * 0.2})`
}

// ── Step 4: Train ──────────────────────────────────────────────────────────────
const trainForm = reactive({
  title: '', predictors: predictorList.map(p => p.key),
  background: 1000, autoOptimize: true, preciseOnly: false,
  regionNorth: null, regionSouth: null, regionEast: null, regionWest: null,
})
const trainError = ref(''); const trainJobId = ref(null)

const trainStatus = computed(() => { const job = maxEnt.activeJob.value; if (!trainJobId.value) return null; return job?.status || null })
const trainStatusLabel = computed(() => {
  const s = trainStatus.value
  if (s === 'succeeded') return 'Training complete ✓'
  if (s === 'failed') return maxEnt.activeJob.value?.error_message || 'Training failed'
  if (s === 'running') return 'Training in progress…'
  return 'Queued…'
})

watch(trainStatus, (s) => { if (s === 'succeeded') completedSteps.value = new Set([...completedSteps.value, 'train']) })

const trainedModel = computed(() => {
  if (!trainJobId.value) return null
  return maxEnt.models.value.find(m => m.id === maxEnt.activeJob.value?.model_configs?.id) || maxEnt.models.value[0] || null
})

async function submitTrain() {
  trainError.value = ''
  const region = (trainForm.regionNorth != null && trainForm.regionSouth != null)
    ? { north: trainForm.regionNorth, south: trainForm.regionSouth, east: trainForm.regionEast, west: trainForm.regionWest }
    : undefined
  const spec = {
    title: trainForm.title, predictors: trainForm.predictors, background: trainForm.background,
    autoOptimize: trainForm.autoOptimize, preciseOnly: trainForm.preciseOnly, region,
    source_dataset_id: enrichDatasetSlug.value || (sourceForm.type === 'dataset' ? sourceForm.datasetSlug : undefined),
    source: buildSource(),
  }
  const result = await maxEnt.trainModel(spec)
  if (!result.ok) { trainError.value = result.error || 'Training submission failed.'; return }
  trainJobId.value = maxEnt.activeJob.value?.job_id
}

// ── Step 5: Evaluate ───────────────────────────────────────────────────────────
const evalData = ref(null); const evalLoading = ref(false); const evalError = ref('')

async function loadEval() {
  const jobId = trainJobId.value || maxEnt.activeJob.value?.job_id
  if (!jobId) return
  evalLoading.value = true; evalError.value = ''
  try {
    const { accessToken } = useAuth()
    const token = await accessToken()
    const headers = token ? { authorization: `Bearer ${token}` } : {}
    const res = await fetch(`/.netlify/functions/modeling-maxent?evaluate=1&jobId=${encodeURIComponent(jobId)}`, { headers })
    const data = await res.json()
    if (!data.ok) throw new Error(data.error || 'Evaluation failed.')
    evalData.value = data
    completedSteps.value = new Set([...completedSteps.value, 'evaluate'])
  } catch (e) {
    evalError.value = e.message
  } finally {
    evalLoading.value = false
  }
}

const aucValue = computed(() => {
  if (evalData.value?.auc != null) return evalData.value.auc
  if (trainedModel.value?.results?.[0]?.auc != null) return trainedModel.value.results[0].auc
  return null
})
const aucDisplay = computed(() => aucValue.value != null ? aucValue.value.toFixed(3) : '—')
const gradeDisplay = computed(() => {
  const g = evalData.value?.grade || trainedModel.value?.results?.[0]?.grade
  return g ? g.charAt(0).toUpperCase() + g.slice(1) : '—'
})
const aucGrade = computed(() => {
  const g = evalData.value?.grade || trainedModel.value?.results?.[0]?.grade
  return g || (aucValue.value >= 0.9 ? 'excellent' : aucValue.value >= 0.75 ? 'good' : aucValue.value >= 0.6 ? 'fair' : 'weak')
})
const contributions = computed(() => {
  const c = evalData.value?.contributions || trainedModel.value?.results?.[0]?.contributions
  if (!c) return []
  return Object.entries(c).map(([key, value]) => ({
    label: PREDICTOR_META[key]?.label || key,
    value: typeof value === 'number' ? value : parseFloat(value),
  })).sort((a, b) => b.value - a.value)
})

function openOnMap() {
  if (!trainedModel.value) return
  modelOverlay.open({ configId: trainedModel.value.id, label: trainedModel.value.title || 'Suitability' })
  router.push('/map')
}

// ── Models tab ─────────────────────────────────────────────────────────────────
const bannerProgress = computed(() => {
  const job = maxEnt.activeJob.value
  if (!job) return 0
  if (job.status === 'succeeded') return 100
  if (job.status === 'failed') return 0
  return 50
})

const registerForm = reactive({ title: '', description: '', asset_path: '', visibility: 'private' })
const registerNote = ref(''); const registerError = ref('')
const visibilities = ['private', 'members', 'public']

async function onRegister() {
  registerNote.value = ''; registerError.value = ''
  const result = await maxEnt.registerAsset({ ...registerForm })
  if (result.ok) {
    registerNote.value = 'Asset registered successfully!'
    registerForm.title = ''; registerForm.description = ''; registerForm.asset_path = ''
  } else {
    registerError.value = result.error || 'Registration failed.'
  }
}

async function confirmDelete(model) {
  if (confirm(`Delete "${model.title}"?`)) {
    const res = await maxEnt.deleteModel(model.id)
    if (res.ok) await maxEnt.fetchModels()
  }
}

function openModelOnMap(model) {
  modelOverlay.open({ configId: model.id, label: model.title || 'Suitability' })
  router.push('/map')
}

function fmtWhen(dateStr) {
  if (!dateStr) return ''
  return new Date(dateStr).toLocaleDateString()
}
</script>

<style scoped>
.model-page { padding: 16px 18px; max-width: 960px; margin: 0 auto; }

/* ── Page tabs ── */
.page-tabs {
  display: flex; gap: 0; margin-bottom: 20px;
  border-bottom: 1px solid var(--border);
}
.ptab {
  padding: 10px 18px; background: none; border: none; border-bottom: 2px solid transparent;
  margin-bottom: -1px; font: inherit; font-size: 0.88rem; font-weight: 600; cursor: pointer;
  color: var(--muted); transition: color 0.12s, border-color 0.12s;
}
.ptab:hover { color: var(--text); }
.ptab.active { color: var(--accent); border-bottom-color: var(--accent); }

/* ── Stepper header ── */
.stepper-header { margin-bottom: 16px; }
.stepper-title-row { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; }
.stepper-title-row h2 { margin: 0 0 4px; font-size: 1.2rem; }
.sub { margin: 0; color: var(--muted); font-size: 0.84rem; }
.mode-toggle {
  flex-shrink: 0; padding: 5px 12px; border: 1px solid var(--border); border-radius: 6px;
  background: var(--surface-2); color: var(--muted); font: inherit; font-size: 0.78rem;
  font-weight: 600; cursor: pointer; white-space: nowrap;
}
.mode-toggle:hover { color: var(--text); border-color: var(--muted); }

/* ── Step indicator ── */
.steps { display: flex; gap: 0; margin-bottom: 20px; border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
.step-btn {
  flex: 1; display: flex; flex-direction: column; align-items: center; gap: 3px;
  padding: 10px 6px; border: 0; border-right: 1px solid var(--border);
  background: var(--surface); cursor: pointer; font-size: 0.78rem; color: var(--muted);
  transition: background 0.15s, color 0.15s;
}
.step-btn:last-child { border-right: 0; }
.step-btn:disabled { cursor: default; opacity: 0.5; }
.step-btn.reachable:hover { background: var(--surface-2); color: var(--text); }
.step-btn.done { color: var(--text); background: var(--surface-2); }
.step-btn.active { background: var(--accent); color: var(--accent-ink); }
.step-num {
  width: 22px; height: 22px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
  background: var(--border); font-size: 0.75rem; font-weight: 700;
}
.step-btn.done .step-num { background: #22c55e; color: #fff; }
.step-btn.active .step-num { background: rgba(0,0,0,0.15); }
.step-label { font-weight: 600; }
@media (max-width: 600px) { .step-label { display: none; } }

/* ── Panel ── */
.panel { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 20px 22px; margin-bottom: 20px; }
.panel h3 { margin: 0 0 6px; font-size: 1.05rem; }
.panel-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; }
.panel-head h3 { margin: 0; font-size: 0.95rem; }
.head-acts { display: flex; align-items: center; gap: 8px; }

/* ── Form rows ── */
.row { display: grid; grid-template-columns: 160px 1fr; gap: 8px 14px; align-items: start; margin-bottom: 14px; }
.row > label:first-child { padding-top: 6px; font-size: 0.84rem; font-weight: 600; color: var(--muted); }
.row input[type="text"], .row input[type="number"], .row select {
  border: 1px solid var(--border); border-radius: 7px; padding: 7px 10px;
  font-size: 0.88rem; background: var(--input-bg); color: var(--text); width: 100%;
}
.opt-label { font-size: 0.75rem; font-weight: 400; color: var(--muted); margin-left: 4px; }
.stack { display: flex; flex-direction: column; gap: 4px; font-size: 0.84rem; }
.stack span { font-weight: 600; color: var(--muted); }
.stack input { border: 1px solid var(--border); border-radius: 7px; padding: 6px 9px; font-size: 0.88rem; background: var(--input-bg); color: var(--text); }
@media (max-width: 600px) { .row { grid-template-columns: 1fr; } }

/* ── Source / stage pickers ── */
.sources, .stages { display: flex; flex-direction: column; gap: 6px; }
.pick { display: flex; align-items: flex-start; gap: 8px; font-size: 0.88rem; cursor: pointer; padding: 6px 10px; border: 1px solid var(--border); border-radius: 7px; background: var(--input-bg); }
.pick:hover { background: var(--surface-2); }
.pick-inline { border: none; background: none; padding: 4px 0; }
.pick-inline:hover { background: none; }
.pick input { margin-top: 2px; }
.stage { padding: 7px 10px; }
.stage span { display: flex; flex-direction: column; }
.stage em { font-size: 0.78rem; color: var(--muted); font-style: normal; }

/* ── Field stack ── */
.field-stack { display: flex; flex-direction: column; gap: 10px; margin-bottom: 12px; }
.field-row { display: grid; grid-template-columns: 120px 1fr; gap: 8px 12px; align-items: start; }
.field-label { padding-top: 6px; font-size: 0.84rem; font-weight: 600; color: var(--muted); }
.date-pair { display: flex; gap: 10px; }
.date-pair .stack { flex: 1; }
@media (max-width: 520px) { .field-row { grid-template-columns: 1fr; } .date-pair { flex-direction: column; } }

/* ── Bbox ── */
.bbox-group { display: flex; flex-direction: column; gap: 8px; }
.bbox { display: flex; gap: 8px; flex-wrap: wrap; }
.mini { display: flex; align-items: center; gap: 4px; font-size: 0.82rem; color: var(--muted); }
.mini input { width: 76px; border: 1px solid var(--border); border-radius: 6px; padding: 5px 7px; font-size: 0.84rem; background: var(--input-bg); color: var(--text); }

/* ── Source tabs ── */
.src-tabs { display: flex; gap: 0; }
.src-tab { border: 1px solid var(--border); background: var(--input-bg); color: var(--muted); padding: 5px 12px; font-size: 0.84rem; cursor: pointer; }
.src-tabs .src-tab:first-child { border-radius: 6px 0 0 6px; }
.src-tabs .src-tab:last-child { border-radius: 0 6px 6px 0; }
.src-tabs .src-tab:not(:first-child) { border-left: none; }
.src-tab.on { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }

/* ── Actions ── */
.actions { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-top: 16px; }
.btn { border: 1px solid var(--border); background: var(--surface-2); border-radius: 7px; padding: 8px 16px; font-size: 0.88rem; font-weight: 600; cursor: pointer; text-decoration: none; color: var(--text); }
.btn.primary { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }
.btn.secondary { background: var(--surface); }
.btn.small { padding: 4px 10px; font-size: 0.78rem; }
.btn.ghost { background: transparent; }
.btn.danger { color: #dc2626; }
.btn:hover:not(:disabled) { opacity: 0.85; }
.btn:disabled { opacity: 0.45; cursor: default; }
.btn-text { background: none; border: none; color: var(--muted); font: inherit; font-size: 0.78rem; cursor: pointer; text-decoration: underline; }
.btn-text:hover { color: var(--text); }
.linkish { background: none; border: none; color: var(--accent); cursor: pointer; font: inherit; font-size: inherit; padding: 0; text-decoration: underline; }

/* ── Job status ── */
.job-status { font-size: 0.85rem; font-weight: 600; padding: 5px 10px; border-radius: 6px; background: var(--surface-2); }
.job-status.succeeded { color: #16a34a; background: #dcfce7; }
.job-status.failed { color: #dc2626; background: #fee2e2; }
.job-status.running, .job-status.pending { color: #d97706; background: #fef3c7; }
.job-progress { margin-top: 14px; }
.prog-bar { height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; margin-bottom: 6px; }
.prog-fill { height: 100%; background: var(--accent); border-radius: 3px; transition: width 0.4s; }
.prog-fill.indeterminate { width: 40%; animation: slide 1.6s infinite; }
@keyframes slide { 0% { transform: translateX(-100%); } 100% { transform: translateX(350%); } }
.prog-msg { font-size: 0.82rem; color: var(--muted); margin: 0; }

/* ── Explore ── */
.explore-section { margin-bottom: 22px; }
.explore-section h4 { margin: 0 0 8px; font-size: 0.92rem; }
.cov-table, .corr-table { width: 100%; border-collapse: collapse; font-size: 0.84rem; }
.cov-table th, .corr-table th { text-align: left; padding: 6px 8px; background: var(--surface-2); border-bottom: 1px solid var(--border); color: var(--muted); font-weight: 600; }
.cov-table td, .corr-table td { padding: 6px 8px; border-bottom: 1px solid var(--border-soft); }
.cov-table tr.low td { color: var(--muted); }
.cov-bar-wrap { display: flex; align-items: center; gap: 8px; }
.cov-bar { display: inline-block; height: 10px; border-radius: 5px; min-width: 2px; }
.cov-pct { font-size: 0.8rem; color: var(--muted); }
.badge { font-size: 0.75rem; font-weight: 600; padding: 2px 7px; border-radius: 4px; }
.badge.ok   { background: #dcfce7; color: #16a34a; }
.badge.warn { background: #fef3c7; color: #d97706; }
.badge.bad  { background: #fee2e2; color: #dc2626; }
.badge-sub { font-size: 0.7rem; font-weight: normal; color: var(--muted); margin-left: 6px; }
.corr-cell { text-align: center; font-size: 0.8rem; min-width: 42px; }
.corr-cell.high { font-weight: 700; }
.row-label { font-weight: 600; font-size: 0.8rem; color: var(--muted); }
.contrib-list { display: flex; flex-direction: column; gap: 5px; margin-top: 8px; }
.contrib-row { display: flex; align-items: center; gap: 8px; }
.contrib-label { width: 130px; font-size: 0.8rem; flex-shrink: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.contrib-bar-wrap { flex: 1; height: 10px; background: var(--border); border-radius: 5px; overflow: hidden; }
.contrib-bar { display: block; height: 100%; background: #2a78d6; border-radius: 5px; }
.contrib-pct { width: 42px; text-align: right; font-size: 0.75rem; color: var(--muted); font-variant-numeric: tabular-nums; }

/* ── Eval summary ── */
.eval-summary { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 16px; }
.eval-stat { background: var(--surface-2); border: 1px solid var(--border); border-radius: 9px; padding: 14px 20px; text-align: center; min-width: 90px; }
.stat-value { display: block; font-size: 1.5rem; font-weight: 700; }
.stat-label { display: block; font-size: 0.76rem; color: var(--muted); margin-top: 2px; }
.eval-stat.excellent .stat-value { color: #16a34a; }
.eval-stat.good      .stat-value { color: #2563eb; }
.eval-stat.fair      .stat-value { color: #d97706; }
.eval-stat.weak      .stat-value { color: #dc2626; }

/* ── Hints / messages ── */
.hint { font-size: 0.82rem; color: var(--muted); margin: 2px 0 10px; }
.hint.warn { color: #d97706; }
.hint.excellent { color: #16a34a; }
.hint.good      { color: #2563eb; }
.hint.fair      { color: #d97706; }
.hint.weak      { color: #dc2626; }
.msg { padding: 10px 14px; border-radius: 7px; font-size: 0.88rem; }
.msg.error { background: #fee2e2; color: #dc2626; }
.msg.ok    { background: #dcfce7; color: #16a34a; }

/* ── Gate ── */
.gate { padding: 32px; text-align: center; }

/* ── Models tab ── */
.head { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; margin-bottom: 20px; }
.head-text h2 { margin: 0 0 4px; }
.head-text .sub { margin: 0; color: var(--muted); font-size: 0.84rem; }
.head-actions { display: flex; align-items: center; gap: 8px; flex-shrink: 0; }

.job-banner {
  display: flex; align-items: flex-start; justify-content: space-between; gap: 12px;
  background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  padding: 12px 16px; margin-bottom: 14px; border-left: 3px solid var(--accent);
}
.banner-body { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 8px; }
.banner-info { display: flex; align-items: center; gap: 10px; }
.banner-name { font-size: 0.88rem; font-weight: 600; }
.banner-bar { height: 5px; background: var(--border); border-radius: 3px; overflow: hidden; }
.bar-fill { height: 100%; background: var(--accent); transition: width 0.4s ease; border-radius: 3px; }
.status-pill {
  font-size: 0.68rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em;
  padding: 2px 7px; border-radius: 999px; border: 1px solid var(--border); color: var(--muted);
}
.status-pill.running, .status-pill.queued { border-color: #3d6b8b; color: #3d6b8b; }
.status-pill.succeeded { border-color: #3d8b5f; color: #3d8b5f; }
.status-pill.failed { border-color: #b3492f; color: #b3492f; }

.loading-msg { padding: 28px 20px; margin: 0; color: var(--muted); font-size: 0.86rem; }
.empty-state { padding: 24px 20px; }
.empty-title { margin: 0 0 4px; font-weight: 600; font-size: 0.95rem; }
.empty-sub { margin: 0 0 14px; color: var(--muted); font-size: 0.84rem; }

.model-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(290px, 1fr)); gap: 1px; background: var(--border); }
.model-card { display: flex; flex-direction: column; gap: 8px; padding: 16px 18px; background: var(--surface); transition: background 0.12s; }
.model-card:hover { background: var(--surface-2); }
.model-card.selected { background: color-mix(in srgb, var(--accent) 6%, var(--surface)); }
.card-top { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.card-check { display: flex; align-items: center; gap: 8px; cursor: pointer; flex: 1; min-width: 0; }
.card-check input[type="checkbox"] { flex-shrink: 0; margin: 0; }
.card-title { font-weight: 600; font-size: 0.9rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.vis-pill {
  font-size: 0.67rem; padding: 2px 7px; border-radius: 999px; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.03em; flex-shrink: 0;
  border: 1px solid var(--border); color: var(--muted);
}
.vis-pill.public { border-color: #3d6b8b; color: #3d6b8b; }
.vis-pill.members { border-color: #3d8b5f; color: #3d8b5f; }
.card-meta { display: flex; align-items: center; flex-wrap: wrap; gap: 4px; font-size: 0.8rem; color: var(--muted); }
.dot { color: var(--border); }
.reg-tag { font-size: 0.72rem; padding: 1px 7px; border-radius: 4px; font-weight: 600; background: color-mix(in srgb, var(--accent) 12%, transparent); color: var(--accent); }
.auc-val { font-weight: 600; }
.auc-val.excellent, .auc-val.good { color: #3d8b5f; }
.auc-val.fair { color: #8a5a1f; }
.auc-val.weak { color: #b3492f; }
.card-footer { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-top: auto; padding-top: 4px; }
.card-date { font-size: 0.74rem; color: var(--muted); }
.card-actions { display: flex; gap: 6px; }

/* ── Advanced panel (link asset) ── */
.adv-panel { border: 1px solid var(--border); border-radius: 10px; margin-top: 12px; overflow: hidden; }
.adv-summary {
  padding: 12px 18px; cursor: pointer; font-size: 0.84rem; font-weight: 600; color: var(--muted);
  background: var(--surface); list-style: none;
}
.adv-summary::-webkit-details-marker { display: none; }
.adv-summary:hover { color: var(--text); }
.adv-body { padding: 16px 18px; border-top: 1px solid var(--border); }
.tab-desc { margin: 0 0 14px; font-size: 0.82rem; color: var(--muted); }
.form-row { display: flex; flex-direction: column; gap: 6px; margin-bottom: 14px; }
.form-row label { font-size: 0.8rem; font-weight: 600; color: var(--muted); }
.form-row.narrow { max-width: 240px; }
.form-row input[type="text"], .form-row select, .form-row textarea {
  padding: 8px 11px; border: 1px solid var(--border); background: var(--surface-2);
  color: var(--text); border-radius: 6px; font: inherit; font-size: 0.86rem;
}
.form-row textarea { resize: vertical; min-height: 56px; }
.form-actions { display: flex; align-items: center; gap: 10px; padding-top: 4px; }
.form-msg { margin: 8px 0 0; font-size: 0.8rem; }
.form-msg.error { color: #b3492f; }
.form-msg.ok { color: #3d8b5f; }

/* ── Comparison overlay ── */
.overlay { position: fixed; inset: 0; z-index: 100; background: rgba(0,0,0,0.6); display: flex; align-items: center; justify-content: center; padding: 20px; }
.overlay-content { background: var(--surface); border-radius: 12px; width: 100%; max-width: 1000px; max-height: 90vh; overflow-y: auto; position: relative; padding: 20px; }
.close-btn { position: absolute; top: 12px; right: 12px; background: var(--surface-2); color: var(--text); border: 1px solid var(--border); border-radius: 50%; width: 26px; height: 26px; cursor: pointer; font-size: 12px; }

@media (max-width: 820px) {
  .model-grid { grid-template-columns: 1fr; }
  .head { flex-direction: column; gap: 10px; }
  .stepper-title-row { flex-direction: column; }
}
</style>
