<template>
  <div class="foray-page">
    <header class="foray-head">
      <div>
        <h2>Plan a foray</h2>
        <p class="sub">Where past finds suggest the species in season are fruiting.</p>
      </div>
      <div class="seg" role="tablist" aria-label="Planner mode">
        <button v-for="m in MODE_LIST" :key="m.key" role="tab" :aria-selected="mode === m.key"
                :class="{ on: mode === m.key }" :title="m.blurb" @click="mode = m.key">{{ m.label }}</button>
      </div>
    </header>

    <section class="near" aria-label="Distance from you">
      <div class="near-row">
        <span class="near-what">Near <strong>{{ origin ? origin.label : 'anywhere' }}</strong></span>
        <label class="near-r">within
          <select v-model.number="radiusMi" :disabled="!origin" aria-label="Distance">
            <option v-for="r in RADIUS_OPTIONS_MI" :key="r" :value="r">{{ radiusLabel(r) }}</option>
          </select>
        </label>
        <label class="near-r">and
          <select v-model.number="driveMin" :disabled="!origin" aria-label="Drive time">
            <option v-for="m in DRIVE_OPTIONS_MIN" :key="m" :value="m">{{ driveLabel(m) }}</option>
          </select>
        </label>
        <div v-if="origin" class="seg seg-sm" role="group" aria-label="Sort places">
          <button :class="{ on: sortBy === 'score' }" :aria-pressed="sortBy === 'score'" @click="sortBy = 'score'">Best</button>
          <button :class="{ on: sortBy === 'near' }" :aria-pressed="sortBy === 'near'" @click="sortBy = 'near'">Nearest</button>
        </div>
        <button v-if="origin && !changing && origin.source !== 'default'" class="link-btn" @click="changing = true">Change</button>
      </div>
      <div v-if="!origin || changing || origin.source === 'default'" class="near-actions">
        <button class="near-btn primary" :disabled="locating" @click="locate()">{{ locating ? 'Locating…' : 'Use my location' }}</button>
        <form class="near-search" @submit.prevent="searchPlace">
          <input v-model="placeText" type="search" placeholder="Town or lat, lon" aria-label="Start from a town or coordinates" enterkeyhint="search">
          <button class="near-btn" type="submit" :disabled="searching || !placeText.trim()">{{ searching ? '…' : 'Go' }}</button>
        </form>
        <button class="near-btn" @click="useMapCentre">Map centre</button>
        <button class="near-btn" :class="{ on: picking }" :aria-pressed="picking" @click="picking = !picking">{{ picking ? 'Tap the map…' : 'Pick on map' }}</button>
        <button v-if="origin" class="link-btn" @click="clearOrigin">Clear</button>
        <button v-if="origin && origin.source !== 'default'" class="link-btn" @click="changing = false; picking = false">Done</button>
      </div>
      <p v-if="locMsg" class="notice warn" role="status">{{ locMsg }}</p>
      <p v-if="origin && driveMin && drive.busy.value" class="notice" role="status">Working out where you can drive in {{ driveLabel(driveMin).replace(' drive', '') }}…</p>
      <p v-else-if="origin && driveMin && drive.message.value" class="notice warn" role="status">{{ drive.message.value }}</p>
      <div class="near-row">
        <span class="near-what">Search</span>
        <TriToggle v-model="builtUp" :options="BUILT_UP_OPTIONS" label="Built-up places" />
      </div>
    </section>

    <details class="filters" :open="filtersOpen" @toggle="filtersOpen = $event.target.open">
      <summary>
        <span class="f-title">Filters</span>
        <span class="f-sum">{{ filterSummary }}</span>
      </summary>
      <section class="foray-controls">
        <label class="ctl">When
          <select v-model="timeSel">
            <option value="now">Now</option>
            <option value="date">On a date</option>
            <option value="range">Between dates</option>
            <option v-for="(m, i) in MONTH_LABELS" :key="m" :value="String(i + 1)">{{ m }}</option>
          </select>
        </label>
        <label v-if="timeSel === 'date' || timeSel === 'range'" class="ctl">{{ timeSel === 'range' ? 'From' : 'Date' }}
          <input v-model="planDate" type="date" :aria-label="timeSel === 'range' ? 'First day' : 'Date to plan for'">
        </label>
        <label v-if="timeSel === 'range'" class="ctl">To
          <input v-model="planEnd" type="date" :min="planDate" aria-label="Last day">
        </label>
        <label class="ctl">Land cover
          <select v-model="landCover">
            <option value="">All</option>
            <option v-for="lc in landCovers" :key="lc" :value="lc">{{ lc }}</option>
          </select>
        </label>
        <label class="ctl">Patch size
          <select v-model.number="cellSize">
            <option v-for="c in SIZE_OPTIONS" :key="c.value" :value="c.value">{{ c.label }}</option>
          </select>
        </label>
        <fieldset class="switches" :disabled="!access.loaded.value">
          <label :title="switchTitle"><input v-model="sw.free" type="checkbox"> Only free places</label>
          <label :title="switchTitle"><input v-model="sw.public" type="checkbox"> Only public lands</label>
          <label :title="switchTitle"><input v-model="sw.collecting" type="checkbox"> Collecting allowed only</label>
          <label :title="switchTitle"><input v-model="sw.includeLikely" type="checkbox" :disabled="!sw.collecting"> Include likely (BLM/USFS, estimated)</label>
          <label v-if="mode !== 'forager'" :title="switchTitle"><input v-model="sw.includeUnknown" type="checkbox"> Keep unknown</label>
        </fieldset>
        <label class="ctl">Walk in
          <select v-model="walkSw.maxWalk" aria-label="Longest walk from parking or a road">
            <option v-for="o in MAX_WALK_OPTIONS" :key="o.value" :value="o.value">{{ o.label }}</option>
          </select>
        </label>
        <label class="chk"><input v-model="walkSw.accessible" type="checkbox"> Accessible parking nearby</label>
        <label class="chk"><input v-model="ediblesOnly" type="checkbox"> Edibles only</label>
      </section>
      <p v-if="planNote" class="notice warn" role="status">{{ planNote }}</p>
      <p v-if="ediblesOnly" class="notice">{{ EDIBLE_DISCLAIMER }}</p>
      <p v-if="walkNote" class="notice" :class="{ warn: walk.message.value }" role="status">{{ walkNote }}</p>
      <p v-if="access.message.value" class="notice warn" role="status">{{ access.message.value }}</p>
      <p v-else-if="anySwitch" class="notice" role="status">
        Unknown fee, land or collecting status is excluded{{ sw.includeUnknown ? ' (except where you chose to keep it)' : '' }}.
      </p>
      <p v-if="sw.includeLikely && sw.collecting" class="notice">{{ LIKELY_DISCLAIMER }}</p>
    </details>

    <details class="filters" :open="tuneOpen" @toggle="tuneOpen = $event.target.open">
      <summary>
        <span class="f-title">Scoring</span>
        <span class="f-sum">Against {{ GRADING_LABELS[params.grading].toLowerCase() }}{{ paramsDefault ? '' : ' · tuned' }}</span>
      </summary>
      <section class="tune">
        <div class="tune-row">
          <span class="tune-l">Grade against</span>
          <div class="seg seg-sm" role="group" aria-label="Grade against">
            <button v-for="g in GRADINGS" :key="g" :class="{ on: params.grading === g }" :aria-pressed="params.grading === g" @click="params.grading = g">{{ GRADING_LABELS[g] }}</button>
          </div>
        </div>
        <p class="notice">{{ gradingText }}</p>
        <label class="tune-s">
          <span><strong>Peak focus</strong> · a species {{ params.halfScale }} days off its peak counts half</span>
          <input v-model.number="params.halfScale" type="range" :min="PARAM_LIMITS.halfScale.min" :max="PARAM_LIMITS.halfScale.max" :step="PARAM_LIMITS.halfScale.step">
        </label>
        <label class="tune-s">
          <span><strong>Time window</strong> · {{ params.window ? `finds within ${params.window} days count` : 'default (14 days for Now, 15 for a month)' }}</span>
          <input v-model.number="params.window" type="range" :min="PARAM_LIMITS.window.min" :max="PARAM_LIMITS.window.max" :step="PARAM_LIMITS.window.step">
        </label>
        <label class="tune-s">
          <span><strong>Small-sample caution</strong> · {{ params.priorFinds ? `treat each place as having ${params.priorFinds} extra average finds` : 'off: every place taken at face value' }}</span>
          <input v-model.number="params.priorFinds" type="range" :min="PARAM_LIMITS.priorFinds.min" :max="PARAM_LIMITS.priorFinds.max" :step="PARAM_LIMITS.priorFinds.step">
        </label>
        <label class="tune-s">
          <span><strong>Fewest finds</strong> · score places with at least {{ params.minSample }} {{ params.minSample === 1 ? 'find' : 'finds' }}</span>
          <input v-model.number="params.minSample" type="range" :min="PARAM_LIMITS.minSample.min" :max="PARAM_LIMITS.minSample.max" :step="PARAM_LIMITS.minSample.step">
        </label>
        <div class="tune-actions">
          <button class="near-btn" :disabled="paramsDefault" @click="resetParams">Reset</button>
          <button class="near-btn" @click="copyLink">{{ linkCopied ? 'Link copied' : 'Copy link to this plan' }}</button>
        </div>
      </section>
    </details>

    <details class="filters" :open="layersOpen" @toggle="onLayersToggle($event.target.open)">
      <summary>
        <span class="f-title">Model layers</span>
        <span class="f-sum">{{ layerSummary }}</span>
      </summary>
      <section class="foray-controls">
        <label class="ctl">Layer
          <select v-model="layerSel">
            <option value="">None</option>
            <option value="ensemble">Model blend for species in season</option>
            <option value="habitat">Habitat match (species in season)</option>
          </select>
        </label>
        <label v-if="layerSel" class="ctl">Opacity
          <input v-model.number="layerOpacity" type="range" min="0.2" max="1" step="0.1" aria-label="Layer opacity">
        </label>
      </section>
      <label v-if="layerSel" class="tune-s">
        <span><strong>Blend into grades</strong> · {{ params.modelMix ? `model influence ${Math.round(params.modelMix * 100)}%` : 'off: the layer is only drawn' }}</span>
        <input v-model.number="params.modelMix" type="range" :min="PARAM_LIMITS.modelMix.min" :max="PARAM_LIMITS.modelMix.max" :step="PARAM_LIMITS.modelMix.step">
      </label>
      <p v-if="layerSel && params.modelMix" class="notice" role="status">
        Places with many finds stay finds-led; places with few lean on the model, up to this share. Promising places with almost no finds join the map as predicted (dashed).
        <template v-if="blendBusy"> Reading the layer at {{ blendTargets.length }} places…</template>
        <template v-else-if="blendOn && blendedCount"> Blended at {{ blendedCount }} places.</template>
      </p>
      <p v-if="blendMsg" class="notice warn" role="status">{{ blendMsg }}</p>
      <p v-if="layers.message.value" class="notice warn" role="status">{{ layers.message.value }}</p>
      <p v-else-if="layers.listed.value && !layers.models.value.length" class="notice">
        No finished models to use yet. Train one on the Models page; public models show here too.
      </p>
      <template v-if="layerSel && layers.models.value.length">
        <p class="notice">{{ layerSel === 'habitat' ? HABITAT_EXPLAINER : ENSEMBLE_EXPLAINER }}</p>
        <ul class="model-list">
          <li v-for="m in layerChoices" :key="m.id">
            <label>
              <input type="checkbox" :checked="pickedIds.includes(m.id)" :disabled="!m.eligible || (!pickedIds.includes(m.id) && pickedIds.length >= MAX_LAYER_MODELS)"
                     @change="togglePick(m.id)">
              <span class="m-title">{{ m.title }}</span>
              <span class="m-meta">{{ m.taxon || 'no taxon' }} · AUC {{ m.auc != null ? m.auc.toFixed(2) : 'not scored' }}<template v-if="!m.own"> · public</template></span>
              <span v-if="m.inSeason" class="chip ok">in season</span>
              <span v-if="!m.eligible" class="chip">{{ layerSel === 'habitat' ? 're-run to add ranges' : 'cannot rebuild' }}</span>
            </label>
          </li>
        </ul>
        <p v-if="layerMeta?.predictors?.length" class="notice">
          Uses {{ layerMeta.predictors.map((p) => `${predictorLabel(p.key)} ${Math.round(p.weight * 100)}%`).join(', ') }}.
        </p>
      </template>
    </details>

    <p v-if="error" class="notice warn">Could not load observations ({{ error }}).</p>
    <p v-else-if="pending && !features.length" class="notice">Loading observations…</p>

    <div class="foray-body">
      <div ref="mapCol" class="map-col">
        <div class="map-box">
          <ClientOnly>
            <ForayMap ref="mapRef" :cells="shown" :ranked="ranked" :selected-key="selectedKey" :overlay="overlayTemplate" :overlay-opacity="layerOpacity"
                      :opportunities="blendOn ? [] : oppCells" :origin="origin" :radius-mi="radiusMi" :drive-area="driveArea" :picking="picking"
                      @select="selectedKey = $event" @view="view = $event" @pick="onPick" />
            <template #fallback><div class="map-fallback">Loading map…</div></template>
          </ClientOnly>
        </div>
        <div class="grad" role="img" :aria-label="`Colour scale from Low to Great, graded against ${GRADING_LABELS[params.grading].toLowerCase()}`">
          <div class="grad-bar" :style="{ background: GRADIENT }">
            <i v-for="x in [25, 50, 75]" :key="x" class="grad-tick" :style="{ left: `${x}%` }"></i>
            <i v-if="sel" class="grad-sel" :style="{ left: `${sel.t * 100}%` }" :title="`Selected place: ${pct(sel.t)}`"></i>
          </div>
          <div class="grad-labels"><span v-for="b in BANDS" :key="b">{{ b }}</span></div>
          <div class="grad-cap">{{ gradeCaption }}<template v-if="sel"> · ▲ selected</template></div>
        </div>
        <div class="legend">
          <span class="lg"><i class="pin">1</i>Top places</span>
          <span v-if="blendOn && scored.some((c) => c.predicted)" class="lg"><i class="pred"></i>Predicted</span>
          <span v-if="opps.ranked.length" class="lg"><i class="pin opp">A</i>Few finds, promising</span>
          <span v-if="overlayTemplate" class="lg"><i class="ramp" :style="{ background: `linear-gradient(90deg, ${SUITABILITY_STOPS.join(',')})` }"></i>{{ layerSel === 'habitat' ? 'Habitat match' : 'Model blend' }} low → high</span>
          <span v-if="origin" class="lg"><i class="me"></i>Start</span>
          <span v-if="driveArea" class="lg"><i class="drive"></i>{{ driveLabel(driveMin) }}</span>
        </div>
        <details class="howto">
          <summary>How to read this map</summary>
          <ul>
            <li>Each dot is a {{ sizeLabel }} patch of ground (zoom in to see its outline) with at least {{ params.minSample }} past iNaturalist {{ params.minSample === 1 ? 'find' : 'finds' }}.</li>
            <li>Its colour is the foray score: how many of the finds there are species that fruit {{ whenText }}, weighted toward species near their peak. Darker green is better; the bar above shows the full scale.</li>
            <li v-if="blendOn">With a model blended in, dashed outlines are predicted places: few or no finds, but the model rates the ground well.</li>
            <li>Numbered pins are the top places in the list below. Tap a pin, dot or list entry to see what was found there.</li>
            <li>No colour means nobody has logged finds there, not that it is poor. Places with only a few finds are ranked cautiously. Open “Why this grade?” on a place to see the working.</li>
          </ul>
        </details>

        <article v-if="sel" class="place" aria-live="polite">
          <div class="p-head">
            <span v-if="selRank" class="p-rank">{{ selRank }}</span>
            <div class="p-title">
              <strong>{{ titleFor(sel) }}</strong>
              <span class="p-coords">{{ sel.lat.toFixed(5) }}, {{ sel.lon.toFixed(5) }} · {{ plusCode(sel.lat, sel.lon) }}</span>
              <span class="p-sub"><template v-if="sel.access.areaName && titleFor(sel) !== sel.access.areaName">{{ sel.access.areaName }} · </template>{{ sel.landCover || 'Land cover unknown' }}<template v-if="sel.distMi != null"> · {{ formatDistance(sel.distMi) }} away</template></span>
            </div>
            <span v-if="sel.predicted" class="chip pred-chip">Predicted</span>
            <span class="band" :class="scoreBand(sel.t).toLowerCase()">{{ scoreBand(sel.t) }}</span>
          </div>
          <p v-if="sel.predicted" class="p-why">
            {{ sel.n === 0 ? 'No finds logged here yet' : `Only ${sel.n} ${sel.n === 1 ? 'find' : 'finds'} here` }}, but the {{ layerSel === 'habitat' ? 'habitat match' : 'model blend' }}
            rates it <strong>{{ pct(sel.promise) }}</strong>. A prompt to look, not a forecast.
          </p>
          <p v-else class="p-why">
            <strong>{{ sel.seasonalN }}</strong> of <strong>{{ sel.n }}</strong> past finds here are species in season {{ whenText }}.
            <template v-if="sel.n < 8"> Only {{ sel.n }} finds, so treat this as a hint.</template>
          </p>
          <ul v-if="sel.components.length" class="p-species">
            <li v-for="x in sel.components.slice(0, 5)" :key="x.species">
              <a :href="taxonUrl(x.species)" target="_blank" rel="noopener"><em>{{ x.species }}</em></a>
              <span v-if="isEdible(x.species)" class="chip ok">edible</span>
              <span class="p-n">{{ x.count }} {{ x.count === 1 ? 'find' : 'finds' }}</span>
            </li>
          </ul>
          <p v-else-if="!sel.predicted" class="p-why">None of the finds here are species in season {{ whenText }}.</p>
          <details class="why">
            <summary>Why this grade?</summary>
            <ol v-if="selWhy" class="why-steps">
              <li>
                <strong>Season score {{ pct(selWhy.raw) }}.</strong>
                {{ selWhy.seasonalN }} of {{ selWhy.n }} finds are in-season species; each counts by how close it is to its peak
                (full weight at peak, half at {{ params.halfScale }} days off).
              </li>
              <li v-if="selWhy.priorFinds">
                <strong>Adjusted to {{ pct(selWhy.adj) }}.</strong>
                With {{ selWhy.n }} finds it is pulled toward the map average ({{ pct(selWhy.mean) }}) as if it had {{ selWhy.priorFinds }} extra average finds.
              </li>
              <li v-if="selWhy.blend">
                <strong>Model blend.</strong>
                The {{ layerSel === 'habitat' ? 'habitat match' : 'model blend' }} here is {{ pct(selWhy.blend.model) }}.
                Finds count {{ pct(selWhy.blend.findsWeight) }} and the model {{ pct(1 - selWhy.blend.findsWeight) }}.
              </li>
              <li>
                <strong>{{ selWhy.band }}.</strong>{{ ' ' }}
                It reaches {{ pct(selWhy.t) }} of the top mark, which is set by the strongest tenth of places {{ GRADING_REF[selWhy.grading] }}<template v-if="selWhy.top"> ({{ pct(selWhy.top * ABSOLUTE_FULL) }} adjusted score)</template>.
                Ranked {{ selWhy.rank }} of {{ selWhy.of }}. Great is 75% and up, Good 50%, Fair 25%.
              </li>
            </ol>
            <ol v-else class="why-steps">
              <li><strong>Predicted.</strong> Too few finds to score from the season, so the grade is the model's {{ pct(sel.promise) }}<template v-if="sel.blend && sel.blend.findsWeight > 0"> with its {{ sel.n }} {{ sel.n === 1 ? 'find' : 'finds' }} counting {{ pct(sel.blend.findsWeight) }}</template>.</li>
              <li><strong>{{ scoreBand(sel.t) }}.</strong> Ranked {{ selGradedRank }} of {{ graded.length }}.</li>
            </ol>
            <table v-if="selWhy && selWhy.species.length" class="why-table">
              <thead><tr><th>Species</th><th>Finds</th><th>Timing</th><th>Weight</th><th>Adds</th></tr></thead>
              <tbody>
                <tr v-for="x in selWhy.species.slice(0, 8)" :key="x.species">
                  <td><em>{{ x.species }}</em></td>
                  <td>{{ x.count }}</td>
                  <td>{{ peakText(x) }}<template v-if="x.iqr != null"> · {{ x.iqr }}-day season</template></td>
                  <td>{{ x.weight.toFixed(2) }}</td>
                  <td>{{ pct(x.contribution) }}</td>
                </tr>
              </tbody>
            </table>
            <p class="notice">Change how this is worked out under Scoring.</p>
          </details>
          <div v-if="sel.walkTier !== 'unknown' || sel.accessible" class="acc">
            <span class="chip" :class="{ ok: sel.walkTier === 'roadside' || sel.walkTier === 'short' }">{{ WALK_LABELS[sel.walkTier] }}</span>
            <span v-if="walkText(sel.walkInfo, sel.walkTier)" class="chip">{{ walkText(sel.walkInfo, sel.walkTier) }}</span>
            <span v-if="sel.accessible" class="chip ok">Accessible parking</span>
            <span v-if="sel.walkInfo?.parking?.fee === 'yes'" class="chip">Paid parking</span>
            <span v-if="sel.walkInfo?.trailhead" class="chip">Trailhead {{ sel.walkInfo.trailhead.name || '' }} {{ (sel.walkInfo.trailhead.m / 1609.344).toFixed(1) }} mi</span>
          </div>
          <div class="acc">
            <span class="chip">{{ sel.access.covered ? sel.access.public_access : 'access unknown' }}</span>
            <span class="chip" :class="{ ok: sel.access.fee_status === 'free' }">fee: {{ feeLabel(sel.access) }}</span>
            <span class="chip" :class="{ ok: sel.access.collecting === 'allowed' }">collecting: {{ collectingLabel(sel.access) }}</span>
          </div>
          <div class="p-links">
            <a :href="directionsUrl(sel)" target="_blank" rel="noopener">{{ sel.walkInfo?.parking ? 'Directions to parking' : 'Directions' }}</a>
            <a :href="inatUrlFor(sel)" target="_blank" rel="noopener">Finds on iNaturalist</a>
            <NuxtLink :to="mapUrlFor(sel)">Show on map page</NuxtLink>
          </div>
        </article>
      </div>

      <div class="list-col">
        <h3 v-if="ranked.length" class="list-h">Top {{ ranked.length }} {{ ediblesOnly ? 'places for edibles' : 'places' }} {{ whenText }}<template v-if="origin && reachText !== 'any distance'"> within {{ reachText }}</template><template v-if="origin && sortBy === 'near'">, nearest first</template></h3>
        <div v-if="!shown.length && !pending" class="empty">
          <template v-if="!base.cells.length">No places with enough finds for this time and land cover{{ ediblesOnly ? ' with edibles in season' : '' }}. Only places that already have finds can be scored.</template>
          <template v-else-if="scored.length && origin && !inReach.length">No scored places within {{ reachText }} of {{ origin.label }}. Try a wider distance or a longer drive.</template>
          <template v-else-if="builtUp !== 'all' && !withWalk.length">No {{ builtUp === 'only' ? 'built-up' : 'non-built-up' }} places in reach. Switch Search to Everywhere.</template>
          <template v-else>No places pass the current filters. Turn a switch off{{ access.loaded.value && !sw.includeUnknown ? ' or keep unknown places' : '' }}.</template>
        </div>

        <p v-if="cfg.showCaveats && shown.length" class="caveat">
          Scores come from past iNaturalist finds, not surveys, and cover only cells with at least {{ params.minSample }} finds<template v-if="blendOn">, plus predicted places from the model blend</template>.
          A blank area is unsampled, not poor. Score is each cell's share of finds that are in-season species, weighted by phenology
          (effort-neutral); ranking pulls small samples toward the average as if each cell had {{ params.priorFinds }} extra average finds.
          It is not a forecast<template v-if="!blendOn"> and does not extrapolate beyond sampled cells</template>.
        </p>

        <ol class="ranked">
          <li v-for="(c, i) in ranked" :key="c.key" :class="{ sel: c.key === selectedKey }" @click="pick(c.key)">
            <div class="row1">
              <span class="rk">{{ i + 1 }}</span>
              <span class="site">
                <strong>{{ titleFor(c) }}</strong>
                <span class="coords">{{ c.lat.toFixed(4) }}, {{ c.lon.toFixed(4) }} · {{ plusCode(c.lat, c.lon) }}</span>
              </span>
              <span v-if="c.distMi != null" class="dist">{{ formatDistance(c.distMi) }}</span>
              <span v-if="c.predicted" class="chip pred-chip">Predicted</span>
              <span class="band" :class="scoreBand(c.t).toLowerCase()">{{ cfg.showComponents ? pct(c.t) : scoreBand(c.t) }}</span>
            </div>
            <div class="row2">
              <span v-if="c.components.length"><em>{{ topNames(c) }}</em></span>
              <span v-else-if="c.predicted">Model rates it {{ pct(c.promise) }}</span>
              <span v-else>Few in-season finds</span>
              <span v-if="c.walkTier !== 'unknown'" class="chip" :class="{ ok: c.walkTier === 'roadside' }">{{ WALK_LABELS[c.walkTier] }}<template v-if="c.accessible"> · ♿</template></span>
              <span>{{ c.n }} finds<template v-if="cfg.showSample"> · {{ density(c) }}/km²</template></span>
              <span v-if="c.landCover && mode === 'forager'">{{ c.landCover }}</span>
              <span v-if="c.access.areaName && titleFor(c) !== c.access.areaName">{{ c.access.areaName }}</span>
              <!-- Only on the picked row: tapping a row is how you ask for more
                   about it, and the same link on all twenty was noise. -->
              <NuxtLink v-if="c.key === selectedKey" class="map-link" :to="mapUrlFor(c)" @click.stop>Show on map</NuxtLink>
              <span v-if="cfg.showAccessCols" class="acc">
                <span class="chip">{{ c.access.covered ? c.access.public_access : 'access unknown' }}</span>
                <span class="chip" :class="{ ok: c.access.fee_status === 'free' }">fee: {{ feeLabel(c.access) }}</span>
                <span v-if="c.access.collecting === 'likely_allowed'" class="chip" :title="LIKELY_DISCLAIMER">{{ LIKELY_LABEL }}</span>
                <span v-else class="chip" :class="{ ok: c.access.collecting === 'allowed' }">collecting: {{ collectingLabel(c.access) }}</span>
              </span>
            </div>
            <div v-if="cfg.showComponents && !c.predicted" class="comp">
              <div>score {{ c.score.toFixed(2) }} = Σ weight × in-window finds ÷ {{ c.n }} finds; sample-adjusted {{ c.adj.toFixed(2) }}<template v-if="c.blend && c.blend.model != null">; model {{ c.blend.model.toFixed(2) }} at {{ pct(1 - c.blend.findsWeight) }}</template>; in-season share {{ Math.round(c.share * 100) }}%; grade position {{ pct(c.t) }}</div>
              <ul>
                <li v-for="x in c.components.slice(0, 4)" :key="x.species">
                  <em>{{ x.species }}</em> · weight {{ x.weight.toFixed(2) }} × {{ x.count }} = {{ x.contribution.toFixed(3) }}
                </li>
              </ul>
            </div>
          </li>
        </ol>
        <p class="notice">{{ DISCLAIMER }}</p>

        <section class="under" aria-labelledby="under-h">
          <div class="sl-head">
            <h3 id="under-h">Where few have looked</h3>
            <div class="sl-actions">
              <button :disabled="!canSearchUnder || underBusy" @click="findUnder">{{ underBusy ? 'Searching…' : 'Search this map view' }}</button>
            </div>
          </div>
          <p class="notice">
            Places on screen with {{ MAX_FINDS }} or fewer past finds where the {{ layerSel === 'habitat' ? 'habitat match' : 'model blend' }} is
            {{ Math.round(MIN_PROMISE * 100) }}% or more, ranked by promise ÷ (1 + finds). Your access filters apply.
            <template v-if="!layerSel"> Pick a model layer above first.</template>
          </p>
          <p v-if="underMsg" class="notice warn" role="status">{{ underMsg }}</p>
          <ol v-if="opps.ranked.length" class="ranked">
            <li v-for="(o, i) in opps.ranked" :key="o.cell.key" :class="{ sel: o.cell.key === selectedKey }" @click="pick(o.cell.key)">
              <div class="row1">
                <span class="rk opp">{{ LETTERS[i] }}</span>
                <strong class="site">{{ titleFor(o.cell) }}</strong>
                <span class="band">{{ Math.round(o.promise * 100) }}%</span>
              </div>
              <div class="row2">
                <span>{{ o.n === 0 ? 'No finds yet' : `${o.n} ${o.n === 1 ? 'find' : 'finds'}` }}</span>
                <span class="acc">
                  <span class="chip">{{ o.cell.access.covered ? o.cell.access.public_access : 'access unknown' }}</span>
                  <span class="chip">collecting: {{ collectingLabel(o.cell.access) }}</span>
                </span>
                <a :href="directionsUrl(o.cell)" target="_blank" rel="noopener" @click.stop>Directions</a>
              </div>
            </li>
          </ol>
          <p v-if="underDone" class="caveat">
            From {{ opps.considered }} under-sampled cells on screen.
            <template v-if="opps.outside"> {{ opps.outside }} more were left out because their habitat is outside the range the models were trained on, so a high score there is a guess.</template>
            Models used: {{ usedModelsText || 'none' }}. A high score is a prompt to look, not a forecast; an empty place may simply lack the habitat.
          </p>
        </section>

        <section v-if="cfg.showShortlist" class="shortlist" id="foray-shortlist">
          <div class="sl-head">
            <h3>Shortlist</h3>
            <div class="sl-actions">
              <ActionMenu title="Take the shortlist with you">
                <button class="mi" :disabled="!rows.length" @click="exportCsv">Export CSV</button>
                <button class="mi" :disabled="!rows.length" @click="copyText">{{ copied ? 'Copied' : 'Copy as text' }}</button>
                <button class="mi" :disabled="!rows.length" @click="printList">Print</button>
              </ActionMenu>
            </div>
          </div>
          <div v-if="rows.length" class="table-wrap">
            <table>
              <thead>
                <tr><th>#</th><th>Site</th><th>Grade</th><th>Access</th><th>Fee</th><th>Collecting</th><th>Walk</th><th>Notes</th></tr>
              </thead>
              <tbody>
                <tr v-for="(r, i) in rows" :key="ranked[i].key">
                  <td>{{ r.rank }}</td>
                  <td>{{ r.near || r.site }}<div class="coords">{{ r.lat }}, {{ r.lon }} · {{ r.plus_code }}</div><div v-if="r.near && ranked[i].access.areaName" class="coords">{{ r.site }}</div></td>
                  <td>{{ r.grade }}</td>
                  <td>{{ r.access_class }}</td>
                  <td>{{ r.fee_status }}</td>
                  <td>{{ r.collecting_status }}</td>
                  <td>{{ r.walk || 'unknown' }}<div v-if="r.parking" class="coords">park {{ r.parking }}</div></td>
                  <td>
                    <input class="note-in" type="text" :value="notes[ranked[i].key] || ''" placeholder="Add a note"
                           :aria-label="`Notes for ${r.site}`" @input="setNote(ranked[i].key, $event.target.value)">
                    <span class="note-print">{{ r.notes }}</span>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          <p v-else class="empty">Nothing to list yet.</p>
          <p class="print-disclaimer">{{ DISCLAIMER }}</p>
        </section>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref, shallowRef, watch } from 'vue'
import { useObservations } from '~/composables/useObservations'
import { useAccess } from '~/composables/useAccess'
import { CELL_SIZES, todayOfYear, useMapHeatmaps } from '~/composables/useMapHeatmaps'
import { rampColor } from '~/composables/ramps'
import {
  ABSOLUTE_FULL, DEFAULT_PARAMS, GRADINGS, GRADING_LABELS, MAX_RANGE_DAYS, MONTH_LABELS, PARAM_LIMITS, adjustForSample, blendScore, daysBetween,
  explainCell, gradeScores, paramsAreDefault, paramsToQuery, parseParams, rankCells, referenceTop, resolvePlan, scoreBand, scoreUnits,
  seasonSampleDays,
} from '~/composables/forayScore'
import {
  MAX_WALK_OPTIONS, WALK_LABELS, driveTarget, isAccessible, passesWalk, walkText, walkTier,
} from '~/composables/forayWalk'
import { useForayWalk } from '~/composables/useForayWalk'
import {
  BUILT_UP_OPTIONS, COLORADO_BBOX, DEFAULT_BUILT_UP, DISCLAIMER, LIKELY_DISCLAIMER, LIKELY_LABEL, FORAY_MODES, attachAccess, buildShortlist,
  collectingLabel, isBuiltUp, effectiveSwitches, feeLabel, filterByAccess, parseBuiltUp, passesBuiltUp, shortlistToCsv,
  shortlistToText, UNKNOWN_ACCESS,
} from '~/composables/forayPlanner'
import {
  DEFAULT_DRIVE_MIN, DEFAULT_RADIUS_MI, DRIVE_OPTIONS_MIN, NO_ORIGIN, RADIUS_OPTIONS_MI, driveLabel, filterByDistance, filterByDrive, formatDistance,
  initialOrigin, parseLatLon, radiusLabel, sortByDistance,
} from '~/composables/forayDistance'
import { useDriveTime } from '~/composables/useDriveTime'
import { EDIBLE_DISCLAIMER, isEdible } from '~/composables/edibles'
import { encodePlusCode as plusCode } from '~/composables/plusCode'
import { usePlaceNames } from '~/composables/usePlaceNames'

import {
  MAX_FINDS, MAX_LAYER_MODELS, MIN_PROMISE, candidateCells, defaultPicks, findsPerCell, modelSpecies, modelsParam, rankUnderSampled,
} from '~/composables/forayModels'
import { useForayLayers } from '~/composables/useForayLayers'
import { MAXENT_PREDICTORS, SUITABILITY_PALETTE } from '~/netlify/lib/maxent.mjs'

useHead({ title: 'Plan a foray · Nexstrata' })

const MODE_LIST = Object.values(FORAY_MODES)
const { data, error, pending, load } = useObservations()
const access = useAccess()
const heat = useMapHeatmaps()

const mode = ref('forager')
const timeSel = ref('now')
const landCover = ref('')
const cellSize = ref(0.02)
const selectedKey = ref('')
const sw = reactive({ ...FORAY_MODES.forager.switches })
const notes = ref({})
const copied = ref(false)
const filtersOpen = ref(false)
const mapCol = ref(null)
const mapRef = ref(null)

// Distance: where "near" is measured from, how far, and the list order.
const origin = ref(null)  // set on mount: saved place, Boulder default, or null if cleared
const radiusMi = ref(DEFAULT_RADIUS_MI)
const sortBy = ref('score')
const changing = ref(false)
const picking = ref(false)
const locating = ref(false)
const searching = ref(false)
const placeText = ref('')
const locMsg = ref('')
// Drive time: places must also be inside the drive-time area from the start
// point, so close-but-roundabout places drop out and far ones are judged by road.
const driveMin = ref(DEFAULT_DRIVE_MIN)
const drive = useDriveTime()
watch([origin, driveMin], ([o, m]) => drive.load(o, m))
const driveArea = computed(() => (origin.value && driveMin.value ? drive.area.value : null))
watch(driveMin, (m) => { try { localStorage.setItem('foray-drive', String(m)) } catch { /* ignore */ } })
// Built-up places: skipped by default, or searched everywhere, or towns only.
const builtUp = ref(DEFAULT_BUILT_UP)
watch(builtUp, (v) => { try { localStorage.setItem('foray-built-up', v) } catch { /* ignore */ } })
// Edibles: score only listed edible species.
const ediblesOnly = ref(false)
watch(ediblesOnly, (v) => { try { localStorage.setItem('foray-edibles', v ? '1' : '0') } catch { /* ignore */ } })
// Planned date or range (for "On a date" / "Between dates").
const isoDay = (d = new Date()) => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
const planDate = ref(isoDay())
const planEnd = ref(isoDay(new Date(Date.now() + 2 * 86400000)))
// Nearest place names for the listed places.
const placeNames = usePlaceNames()

// Scoring controls; shared through the URL so a tuned plan can be sent as a link.
const params = reactive({ ...DEFAULT_PARAMS })
const tuneOpen = ref(false)
const linkCopied = ref(false)
const paramsDefault = computed(() => paramsAreDefault({ ...params }))
function resetParams() { Object.assign(params, DEFAULT_PARAMS) }
watch(params, (p) => {
  if (typeof window === 'undefined') return
  const u = new URL(window.location.href)
  for (const k of ['fh', 'fp', 'fm', 'fw', 'fg', 'fx']) u.searchParams.delete(k)
  for (const [k, v] of Object.entries(paramsToQuery({ ...p }))) u.searchParams.set(k, v)
  window.history.replaceState(window.history.state, '', u)
})
async function copyLink() {
  try {
    await navigator.clipboard.writeText(window.location.href)
    linkCopied.value = true
    setTimeout(() => { linkCopied.value = false }, 1500)
  } catch { /* clipboard blocked */ }
}

// Walk-in: parking and road distance per place, and the guide's filter.
const walk = useForayWalk()
const walkSw = reactive({ maxWalk: 'any', accessible: false })
const walkActive = computed(() => walkSw.maxWalk !== 'any' || walkSw.accessible)

const SIZE_OPTIONS = CELL_SIZES.filter((x) => x.value >= 0.01 && x.value <= 0.1)
// Readable on the light OSM base: the low end is a visible yellow-green, not near-white.
const ramp = ['#e6f598', '#a6d96a', '#41ab5d', '#00592b']
const GRADIENT = `linear-gradient(90deg, ${ramp.join(', ')})`
const BANDS = ['Low', 'Fair', 'Good', 'Great']

const cfg = computed(() => FORAY_MODES[mode.value])

// Mode only changes defaults and which panels show; the score is shared.
watch(mode, (m) => {
  Object.assign(sw, FORAY_MODES[m].switches)
  try { localStorage.setItem('foray-mode', m) } catch { /* ignore */ }
})

onMounted(async () => {
  try {
    const m = localStorage.getItem('foray-mode')
    if (m && FORAY_MODES[m]) mode.value = m
    notes.value = JSON.parse(localStorage.getItem('foray-notes') || '{}') || {}
    origin.value = initialOrigin(localStorage.getItem('foray-origin'))
    const r = Number(localStorage.getItem('foray-radius'))
    if (RADIUS_OPTIONS_MI.includes(r) && localStorage.getItem('foray-radius') !== null) radiusMi.value = r
    if (localStorage.getItem('foray-sort') === 'near') sortBy.value = 'near'
    const dm = localStorage.getItem('foray-drive')
    if (dm !== null && DRIVE_OPTIONS_MIN.includes(Number(dm))) driveMin.value = Number(dm)
    builtUp.value = parseBuiltUp(localStorage.getItem('foray-built-up'))
    ediblesOnly.value = localStorage.getItem('foray-edibles') === '1'
  } catch { origin.value = initialOrigin(null) }
  placeNames.restore()
  // Ask for the user's location on load unless they picked a place or cleared it.
  // If they decline, the Boulder default (or last GPS fix) stays, with no warning.
  if (origin.value && (origin.value.source === 'default' || origin.value.source === 'gps')) locate({ quiet: true })
  const search = new URLSearchParams(location.search)
  const q = search.get('mode')
  if (q && FORAY_MODES[q]) mode.value = q
  Object.assign(params, parseParams(search))
  if (!paramsAreDefault({ ...params })) tuneOpen.value = true
  await load()
  access.load(COLORADO_BBOX)
})

const features = computed(() => data.value?.features || [])

const landCovers = computed(() => {
  const n = new Map()
  for (const f of features.value) {
    const l = f.properties?.land_cover_label
    if (l) n.set(l, (n.get(l) || 0) + 1)
  }
  return [...n.entries()].sort((a, b) => b[1] - a[1]).map(([k]) => k)
})

const plan = computed(() => {
  const t = timeSel.value
  if (t === 'now' || t === 'date' || t === 'range') return { kind: t, date: planDate.value, end: planEnd.value }
  return { kind: 'month', month: Number(t) }
})
const time = computed(() => resolvePlan(plan.value, todayOfYear(), params.window || 14, params.window || 15))
const planNote = computed(() => {
  if (timeSel.value !== 'range') return ''
  const span = daysBetween(planDate.value, planEnd.value)
  if (span != null && span > 0 && span <= MAX_RANGE_DAYS) return ''
  return `Pick a last day after the first and within ${MAX_RANGE_DAYS} days. Planning on the first day for now.`
})

const speciesFilter = computed(() => (ediblesOnly.value ? isEdible : undefined))
const base = computed(() => heat.computeForayCells(features.value, {
  day: time.value.day, window: time.value.window,
  landCover: landCover.value || undefined, size: cellSize.value,
  halfScale: params.halfScale, minSample: params.minSample, speciesFilter: speciesFilter.value,
}))

// ── Grade scale ────────────────────────────────────────────────────────────
// A grade is a share of a top mark: the 90th percentile of reference scores.
// "This date" uses the places on the planned date; "Whole season" samples the
// fruiting season at ten days; "All past seasons" does the same for each past
// year on its own finds. The latter two take a moment, so they run after paint.
const GRADING_REF = {
  now: 'in the region on this date',
  season: 'across the whole fruiting season',
  history: 'across every past season, year by year',
}
const refTop = shallowRef({ key: '', top: null })
const refBusy = ref(false)
const refKey = computed(() => (params.grading === 'now' || !features.value.length ? '' : JSON.stringify([
  params.grading, time.value.window, landCover.value, cellSize.value, params.halfScale, params.minSample, params.priorFinds,
  ediblesOnly.value, features.value.length,
])))
function yearOf(p) {
  const y = Number(p?.year ?? String(p?.date || '').slice(0, 4))
  return Number.isInteger(y) && y > 1900 ? y : null
}
function computeReference() {
  const feats = features.value
  const days = seasonSampleDays(feats.map((f) => Number(f.properties?.day_of_year)).filter(Number.isFinite), 10)
  let groups = [feats]
  if (params.grading === 'history') {
    const thisYear = new Date().getFullYear()
    const by = new Map()
    for (const f of feats) {
      const y = yearOf(f.properties)
      if (y == null || y >= thisYear) continue
      if (!by.has(y)) by.set(y, [])
      by.get(y).push(f)
    }
    groups = [...by.values()].filter((g) => g.length >= 50)
  }
  const values = []
  for (const day of days) {
    for (const g of groups) {
      const { cells } = heat.computeForayCells(g, {
        day, window: time.value.window, landCover: landCover.value || undefined, size: cellSize.value,
        halfScale: params.halfScale, minSample: params.minSample, speciesFilter: speciesFilter.value, seasonFrom: feats,
      })
      for (const c of adjustForSample(cells, params.priorFinds)) values.push(scoreUnits(c.adj))
    }
  }
  return referenceTop(values)
}
let refTimer = null
watch(refKey, (k) => {
  clearTimeout(refTimer)
  if (!k || refTop.value.key === k) { refBusy.value = false; return }
  refBusy.value = true
  refTimer = setTimeout(() => {
    if (k !== refKey.value) return
    try { refTop.value = { key: k, top: computeReference() } } finally { refBusy.value = false }
  }, 30)
}, { immediate: true })
// null grades the places against themselves (This date, or while the scale is worked out).
const gradeTop = computed(() => (params.grading !== 'now' && refTop.value.key === refKey.value ? refTop.value.top : null))
const gradingText = computed(() => {
  const g = params.grading
  const text = g === 'now'
    ? 'Graded against the other places in the region on the planned date, so the strongest that day read Great, even late in the season.'
    : g === 'season'
      ? 'Graded against how places score across the whole fruiting season, so late or early in the season most places read lower.'
      : 'Graded against every past season, year by year, good years and poor ones.'
  if (g === 'now') return text
  if (refBusy.value) return `${text} Working out the scale…`
  if (refTop.value.key === refKey.value && refTop.value.top == null) return `${text} Not enough finds to set this scale, so places are graded against this date instead.`
  return text
})
const gradeCaption = computed(() => `Graded against ${GRADING_LABELS[params.grading].toLowerCase()}`)

const effective = computed(() => effectiveSwitches({ ...sw }, access.status.value))
const anySwitch = computed(() => effective.value.free || effective.value.public || effective.value.collecting)
const switchTitle = computed(() => (access.loaded.value ? '' : 'Needs access data for this area'))

// Finds-scored cells after the access filter, with the sample adjustment.
const adjusted = computed(() => {
  const withAccess = access.loaded.value || access.status.value === 'partial'
    ? attachAccess(base.value.cells, access.areas.value)
    : base.value.cells.map((c) => ({ ...c, access: { ...UNKNOWN_ACCESS } }))
  return adjustForSample(filterByAccess(withAccess, effective.value), params.priorFinds)
})
// Every graded cell. `final` is on the absolute scale (scoreUnits): the finds
// score alone, or blended with the model where a layer is mixed in. With the
// blend on, the promising "few have looked" places join as predicted cells.
const scored = computed(() => {
  const mix = blendOn.value ? params.modelMix : 0
  const cells = adjusted.value.map((c) => {
    const b = mix ? blendScore(c.adj, c.n, modelVals.value.get(c.key)?.value, mix) : null
    return { ...c, blend: b, final: b ? b.blend : scoreUnits(c.adj) }
  })
  if (mix) {
    const have = new Set(cells.map((c) => c.key))
    for (const o of opps.value.ranked) {
      if (have.has(o.cell.key)) continue
      const b = blendScore(0, o.n, o.promise, mix)
      cells.push({
        ...o.cell, n: o.n, seasonalN: 0, score: 0, share: 0, adj: 0, thin: false, components: [], landCover: null,
        predicted: true, promise: o.promise, blend: b, final: b.blend,
      })
    }
  }
  return gradeScores(cells, 'final', gradeTop.value)
})
// Best-first over the whole region, for the rank in "Why this grade?".
const graded = computed(() => rankCells(scored.value, Infinity, 'final'))
// Scores and colours are graded over the whole region before the distance
// cut, so a "Great" nearby place is great statewide, not just the best of a few.
// The walk filter is applied after it for the same reason.
const near = computed(() => filterByDrive(filterByDistance(scored.value, origin.value, radiusMi.value), driveArea.value))
const inReach = near
const reachText = computed(() => [
  ...(origin.value && radiusMi.value ? [`${radiusMi.value} mi`] : []),
  ...(driveArea.value ? [driveLabel(driveMin.value)] : []),
].join(' and ') || 'any distance')
const withWalk = computed(() => near.value
  .filter((c) => passesBuiltUp(isBuiltUp(c), builtUp.value))
  .map((c) => {
  const info = walk.info.value.get(c.key) || null
  const tier = walkTier(info, access.loaded.value)
  return { ...c, walkInfo: info, walkTier: tier, accessible: isAccessible(info) }
}))
const shown = computed(() => withWalk.value
  .filter((c) => passesWalk(c.walkTier, c.accessible, { ...walkSw, keepUnknown: sw.includeUnknown }))
  .map((c) => ({
    ...c,
    color: rampColor(ramp, c.t),
    tip: `${c.predicted ? 'Predicted · ' : ''}${scoreBand(c.t)} · ${c.n} finds${c.distMi != null ? ` · ${formatDistance(c.distMi)}` : ''}${c.walkTier !== 'unknown' ? ` · ${WALK_LABELS[c.walkTier]}` : ''}${c.components.length ? ` · ${c.components[0].species}` : ''}`,
  })))
// Nearest re-orders the best places in range; it does not promote poor ones.
const ranked = computed(() => {
  const top = rankCells(shown.value, cfg.value.listLimit, 'final')
  return origin.value && sortBy.value === 'near' ? sortByDistance(top) : top
})
const rows = computed(() => buildShortlist(ranked.value.map((c) => ({
  ...c,
  placeName: placeNames.nameAt(c.lat, c.lon),
  grade: `${c.predicted ? 'Predicted ' : ''}${scoreBand(c.t)}`,
  walk: c.walkTier === 'unknown' ? '' : `${WALK_LABELS[c.walkTier]}${walkText(c.walkInfo, c.walkTier) ? ` · ${walkText(c.walkInfo, c.walkTier)}` : ''}`,
  parking: c.walkInfo?.parking || null,
})), notes.value, cfg.value.listLimit))

const walkNote = computed(() => {
  if (walk.message.value) return walk.message.value
  if (!walkActive.value) return ''
  const unknown = withWalk.value.filter((c) => c.walkTier === 'unknown').length
  if (walk.busy.value) return 'Looking up parking and roads…'
  return unknown ? `${unknown} place${unknown === 1 ? ' has' : 's have'} no parking data yet and ${sw.includeUnknown ? 'are kept' : 'are hidden'}${mode.value !== 'forager' ? ' (Keep unknown)' : ''}.` : ''
})

// ── Model layers (WANT-17 phases 2 and 3) ───────────────────────────────────
const ENSEMBLE_EXPLAINER = 'A weighted blend of the suitability maps of your models for species in season, weighted by how close each species is to its peak. Each model is refit, so it can take a minute.'
const HABITAT_EXPLAINER = 'Shades ground whose terrain, climate and cover sit in the range where the in-season models\' finds were made, using the variables that mattered most to them. Not species-specific, and only as good as those models.'
const SUITABILITY_STOPS = SUITABILITY_PALETTE
const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
const layers = useForayLayers()
const layersOpen = ref(false)
const layerSel = ref('')
const layerOpacity = ref(0.6)
const pickedIds = ref([])
const overlayTemplate = ref(null)
const view = ref(null)
const layerMeta = ref(null)
const predictorLabel = (k) => MAXENT_PREDICTORS[k]?.label || k

function onLayersToggle(open) {
  layersOpen.value = open
  if (open && !layers.listed.value) layers.list()
}
watch(layerSel, (v) => { if (v && !layers.listed.value) layers.list() })

const seasonWeightsMap = computed(() => base.value.weights || new Map())
const defaults = computed(() => (layerSel.value ? defaultPicks(layers.models.value, seasonWeightsMap.value, layerSel.value) : []))
const layerChoices = computed(() => {
  const names = [...seasonWeightsMap.value.keys()]
  return layers.models.value
    .map((m) => ({
      ...m,
      inSeason: modelSpecies(m, names).length > 0,
      eligible: layerSel.value === 'habitat' ? m.ranges && m.contributions : m.usable,
    }))
    .sort((a, b) => Number(b.inSeason) - Number(a.inSeason) || a.title.localeCompare(b.title))
})
// The picks follow the season and the layer until changed by hand.
watch(defaults, (d) => { pickedIds.value = d.map((p) => p.id) }, { immediate: true })
function togglePick(id) {
  pickedIds.value = pickedIds.value.includes(id) ? pickedIds.value.filter((x) => x !== id) : [...pickedIds.value, id].slice(0, MAX_LAYER_MODELS)
}
// A hand-picked model outside the season gets a modest weight.
const picks = computed(() => pickedIds.value.map((id) => defaults.value.find((p) => p.id === id) || { id, weight: 0.5, species: [] }))
const picksParam = computed(() => modelsParam(picks.value))

let mintSeq = 0
watch([layerSel, picksParam], async ([layer, param]) => {
  const seq = ++mintSeq
  overlayTemplate.value = null
  layerMeta.value = null
  opps.value = { ranked: [], outside: 0, considered: 0 }
  underDone.value = false
  if (!layer || !param) return
  const r = await layers.mint(layer, param)
  if (seq !== mintSeq || !r) return
  overlayTemplate.value = r.template
  layerMeta.value = r.meta
})

// ── Model blend into the score ─────────────────────────────────────────────
// The layer is sampled at each place in range (best first, capped) and mixed
// into its grade by params.modelMix; places with few finds lean on the model.
const BLEND_BATCH = 1500
const BLEND_MAX = 3000
const blendOn = computed(() => Boolean(layerSel.value && overlayTemplate.value && params.modelMix > 0))
const modelVals = shallowRef(new Map())
const blendBusy = ref(false)
const blendMsg = ref('')
let blendSeq = 0
watch(picksParam, () => { modelVals.value = new Map() })
watch(layerSel, () => { modelVals.value = new Map() })
async function sampleModel(cells) {
  const want = cells.filter((c) => !c.predicted && !modelVals.value.has(c.key)).slice(0, BLEND_MAX)
  if (!want.length) return
  const seq = ++blendSeq
  blendBusy.value = true
  blendMsg.value = ''
  try {
    for (let i = 0; i < want.length; i += BLEND_BATCH) {
      const part = want.slice(i, i + BLEND_BATCH)
      const res = await layers.sample(layerSel.value, picksParam.value, part.map((c) => [c.lon, c.lat]))
      if (seq !== blendSeq) return
      if (!res) { blendMsg.value = layers.message.value || 'Could not read the model layer here.'; return }
      const next = new Map(modelVals.value)
      part.forEach((c, j) => next.set(c.key, { value: res.samples[j]?.value ?? null, outside: res.samples[j]?.outside ?? null }))
      modelVals.value = next
    }
  } finally {
    if (seq === blendSeq) blendBusy.value = false
  }
}
const blendTargets = computed(() => (blendOn.value
  ? rankCells(filterByDrive(filterByDistance(adjusted.value, origin.value, radiusMi.value), driveArea.value), Infinity, 'adj')
  : []))
watch(blendTargets, (cells) => { if (cells.length) sampleModel(cells) })
// Turning the blend on also looks for promising unvisited places in view, once.
let autoUnder = ''
watch([blendOn, overlayTemplate, view], () => {
  if (!blendOn.value || !view.value || underBusy.value) return
  const tag = `${layerSel.value}|${picksParam.value}`
  if (autoUnder === tag) return
  autoUnder = tag
  findUnder()
})
const blendedCount = computed(() => adjusted.value.filter((c) => modelVals.value.get(c.key)?.value != null).length)

const layerSummary = computed(() => {
  if (!layerSel.value) return 'None'
  const n = pickedIds.value.length
  return `${layerSel.value === 'habitat' ? 'Habitat match' : 'Model blend'} · ${n} ${n === 1 ? 'model' : 'models'}`
})

// ── Where few have looked (WANT-17 phase 5) ─────────────────────────────────
const underBusy = ref(false)
const underMsg = ref('')
const underDone = ref(false)
const opps = ref({ ranked: [], outside: 0, considered: 0 })
const oppCells = computed(() => opps.value.ranked.map((o) => o.cell))
const canSearchUnder = computed(() => Boolean(layerSel.value && pickedIds.value.length && view.value))
const usedModelsText = computed(() => {
  const used = new Set(layerMeta.value?.used || pickedIds.value)
  return layers.models.value.filter((m) => used.has(m.id))
    .map((m) => `${m.title} (AUC ${m.auc != null ? m.auc.toFixed(2) : 'not scored'})`).join(', ')
})

async function findUnder() {
  underMsg.value = ''
  underDone.value = false
  const shape = heat.cellShape.value
  const { cells, tooMany } = candidateCells(view.value, cellSize.value, shape)
  if (tooMany) { underMsg.value = 'Zoom in closer (or pick a bigger patch size) to search this view.'; return }
  const finds = findsPerCell(features.value, cellSize.value, shape, landCover.value || undefined)
  const few = cells.map((c) => ({ ...c, n: finds.get(c.key) || 0 })).filter((c) => c.n <= MAX_FINDS)
  const withAccess = access.loaded.value || access.status.value === 'partial'
    ? attachAccess(few, access.areas.value)
    : few.map((c) => ({ ...c, access: { ...UNKNOWN_ACCESS } }))
  const candidates = filterByAccess(withAccess, effective.value)
  if (!candidates.length) { underMsg.value = 'No under-sampled cells on screen pass your filters.'; opps.value = { ranked: [], outside: 0, considered: 0 }; return }
  underBusy.value = true
  try {
    const res = await layers.sample(layerSel.value, picksParam.value, candidates.map((c) => [c.lon, c.lat]))
    if (!res) { underMsg.value = layers.message.value; return }
    const merged = candidates.map((c, i) => ({ ...c, promise: res.samples[i]?.value ?? null, outside: res.samples[i]?.outside ?? null }))
    opps.value = rankUnderSampled(merged, { limit: cfg.value.listLimit })
    underDone.value = true
    if (!opps.value.ranked.length) underMsg.value = 'Nothing on screen scores high enough. Try another view or layer.'
  } finally {
    underBusy.value = false
  }
}

const sel = computed(() => shown.value.find((c) => c.key === selectedKey.value) || null)
const selRank = computed(() => ranked.value.findIndex((c) => c.key === selectedKey.value) + 1)
// Keep a place selected so the details card always has something to say.
watch(ranked, (r) => {
  if (!sel.value && r.length) selectedKey.value = r[0].key
})

// Look up parking for the listed places (and the selected one); for every place
// in range only while a walk filter needs it.
watch([near, ranked, walkActive], () => {
  const want = walkActive.value
    ? rankCells(near.value, Infinity, 'final')
    : [...ranked.value, ...(sel.value ? [sel.value] : [])]
  walk.refresh(want.filter((c) => !c.predicted || walkActive.value))
})
const selWhy = computed(() => (sel.value && !sel.value.predicted
  ? explainCell(sel.value, { cells: adjusted.value, ranked: graded.value, species: base.value.species || [], params, top: gradeTop.value })
  : null))
// Names for the listed places, best first, then the selected one and the "few have looked" ones.
watch([ranked, sel, () => opps.value.ranked], () => {
  placeNames.request([...ranked.value, ...(sel.value ? [sel.value] : []), ...opps.value.ranked.map((o) => o.cell)])
})
const coordText = (c) => `${c.lat.toFixed(4)}, ${c.lon.toFixed(4)}`
const titleFor = (c) => placeNames.nameAt(c.lat, c.lon) || c.access?.areaName || coordText(c)
const mapUrlFor = (c) => `/map?c=${c.lat.toFixed(5)},${c.lon.toFixed(5)}&z=14`
const selGradedRank = computed(() => (sel.value ? graded.value.findIndex((c) => c.key === sel.value.key) + 1 : 0))
const pct = (v) => `${Math.round((Number(v) || 0) * 100)}%`
const peakText = (x) => (x.dist == null ? '' : x.dist === 0 ? 'at peak' : `${x.dist} days off peak`)

function setOrigin(o) {
  origin.value = o
  changing.value = false
  picking.value = false
  locMsg.value = ''
  try { localStorage.setItem('foray-origin', JSON.stringify(o)) } catch { /* ignore */ }
}
function clearOrigin() {
  origin.value = null
  changing.value = false
  picking.value = false
  try { localStorage.setItem('foray-origin', NO_ORIGIN) } catch { /* ignore */ }
}
watch(radiusMi, (r) => { try { localStorage.setItem('foray-radius', String(r)) } catch { /* ignore */ } })
watch(sortBy, (v) => { try { localStorage.setItem('foray-sort', v) } catch { /* ignore */ } })

function locate({ quiet = false } = {}) {
  if (!('geolocation' in navigator)) {
    if (!quiet) locMsg.value = 'This browser cannot share your location. Type a town, use the map centre or pick a spot on the map.'
    return
  }
  locating.value = true
  locMsg.value = ''
  navigator.geolocation.getCurrentPosition(
    (pos) => {
      locating.value = false
      setOrigin({ lat: pos.coords.latitude, lon: pos.coords.longitude, label: 'your location', source: 'gps' })
    },
    (err) => {
      locating.value = false
      if (!quiet) locMsg.value = `${err.code === err.PERMISSION_DENIED ? 'Location permission denied.' : 'Could not get your location.'} Type a town, use the map centre or pick a spot on the map.`
    },
    { enableHighAccuracy: false, timeout: 10000, maximumAge: 300000 },
  )
}

function useMapCentre() {
  const c = mapRef.value?.getCenter?.()
  if (c) setOrigin({ lat: c[0], lon: c[1], label: 'map centre', source: 'map' })
}

function onPick([lat, lon]) {
  setOrigin({ lat, lon, label: `${lat.toFixed(3)}, ${lon.toFixed(3)}`, source: 'map' })
}

// Typed coordinates are used as-is; anything else goes to OpenStreetMap's geocoder, biased to Colorado.
async function searchPlace() {
  const q = placeText.value.trim()
  if (!q) return
  const ll = parseLatLon(q)
  if (ll) { setOrigin({ ...ll, label: q, source: 'typed' }); return }
  searching.value = true
  locMsg.value = ''
  try {
    const [w, s, e, n] = COLORADO_BBOX
    const url = `https://nominatim.openstreetmap.org/search?format=jsonv2&limit=1&countrycodes=us&viewbox=${w},${n},${e},${s}&q=${encodeURIComponent(q)}`
    const res = await fetch(url, { headers: { Accept: 'application/json' } })
    const hit = res.ok ? (await res.json())?.[0] : null
    if (hit) setOrigin({ lat: Number(hit.lat), lon: Number(hit.lon), label: hit.name || String(hit.display_name || q).split(',')[0], source: 'typed' })
    else locMsg.value = `Could not find “${q}”. Try a nearby town or “lat, lon”.`
  } catch {
    locMsg.value = 'Place search is unavailable right now. Try “lat, lon”, the map centre or pick a spot on the map.'
  } finally {
    searching.value = false
  }
}

const sizeLabel = computed(() => SIZE_OPTIONS.find((c) => c.value === cellSize.value)?.label || 'small')
const whenText = computed(() => {
  if (timeSel.value === 'date') return `around ${time.value.label}`
  if (timeSel.value === 'range') return time.value.label === 'Now' ? 'now' : `for ${time.value.label}`
  if (timeSel.value !== 'now') return `around mid-${time.value.label}`
  const d = new Date()
  return `within ${time.value.window} days of ${d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}`
})
const filterSummary = computed(() => [
  timeSel.value === 'now' ? 'Now' : time.value.label,
  ...(ediblesOnly.value ? ['edibles'] : []),
  landCover.value || 'All land cover',
  sizeLabel.value,
  ...(anySwitch.value ? ['access filters on'] : []),
  ...(walkSw.maxWalk !== 'any' ? [MAX_WALK_OPTIONS.find((o) => o.value === walkSw.maxWalk).label] : []),
  ...(walkSw.accessible ? ['accessible parking'] : []),
].join(' · '))

function pick(key) {
  selectedKey.value = key
  // On a phone the map is above the list; bring it back into view.
  const el = mapCol.value
  if (el && window.matchMedia('(max-width: 800px)').matches && el.getBoundingClientRect().top < 0) {
    el.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }
}

const directionsUrl = (c) => {
  const d = driveTarget(c, c.walkInfo)
  return `https://www.google.com/maps/dir/?api=1&destination=${d.lat.toFixed(5)},${d.lon.toFixed(5)}`
}
const inatUrlFor = (c) => {
  const km = Math.max(1, Math.round(cellSize.value * 111 * 0.75))
  return `https://www.inaturalist.org/observations?lat=${c.lat.toFixed(4)}&lng=${c.lon.toFixed(4)}&radius=${km}&iconic_taxa=Fungi`
}
const taxonUrl = (name) => `https://www.inaturalist.org/taxa/search?q=${encodeURIComponent(name)}`

const topNames = (c) => {
  const names = c.components.slice(0, 2).map((x) => x.species).join(', ')
  return c.components.length > 2 ? `${names} +${c.components.length - 2} more` : names
}
const density = (c) => {
  const km2 = (cellSize.value * 111) ** 2 * Math.cos((c.lat * Math.PI) / 180)
  return (c.n / km2).toFixed(1)
}

function setNote(key, v) {
  notes.value = { ...notes.value, [key]: v }
  try { localStorage.setItem('foray-notes', JSON.stringify(notes.value)) } catch { /* ignore */ }
}

function exportCsv() {
  const blob = new Blob([shortlistToCsv(rows.value)], { type: 'text/csv;charset=utf-8' })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = `foray-shortlist-${new Date().toISOString().slice(0, 10)}.csv`
  a.click()
  URL.revokeObjectURL(a.href)
}

async function copyText() {
  try {
    await navigator.clipboard.writeText(shortlistToText(rows.value))
    copied.value = true
    setTimeout(() => { copied.value = false }, 1500)
  } catch { /* clipboard blocked */ }
}

const printList = () => window.print()
</script>

<style scoped>
.foray-page { box-sizing: border-box; height: 100%; overflow-y: auto; padding: 0.8rem 1rem 1.5rem; display: flex; flex-direction: column; gap: 0.6rem; color: var(--text); }
.foray-head { display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 0.6rem; }
h2 { margin: 0; font-size: 1.2rem; color: var(--text-strong); }
.sub { margin: 0.1rem 0 0; font-size: 0.82rem; color: var(--muted); }
.seg { display: inline-flex; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
.seg button { font: inherit; font-size: 0.82rem; padding: 0.35rem 0.8rem; background: var(--surface); color: var(--text); border: 0; cursor: pointer; }
.seg button + button { border-left: 1px solid var(--border); }
.seg button.on { background: var(--accent); color: var(--accent-ink, #fff); font-weight: 600; }

.near { border: 1px solid var(--border-soft); border-radius: 8px; padding: 0.45rem 0.7rem; background: var(--surface); display: flex; flex-direction: column; gap: 0.45rem; }
.near-row { display: flex; flex-wrap: wrap; align-items: center; gap: 0.4rem 0.8rem; font-size: 0.88rem; }
.near-what { min-width: 0; overflow-wrap: anywhere; }
.near-r { display: inline-flex; align-items: center; gap: 0.3rem; color: var(--muted); }
.near-r select { font: inherit; font-size: 0.85rem; background: var(--input-bg, var(--surface)); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 0.25rem 0.4rem; }
.seg-sm button { font-size: 0.78rem; padding: 0.25rem 0.6rem; }
.near-actions { display: flex; flex-wrap: wrap; align-items: center; gap: 0.4rem; }
.near-btn { font: inherit; font-size: 0.8rem; padding: 0.35rem 0.7rem; border: 1px solid var(--border); border-radius: 6px; background: var(--surface-2); color: var(--text); cursor: pointer; }
.near-btn.primary, .near-btn.on { background: var(--accent); border-color: var(--accent); color: var(--accent-ink, #fff); font-weight: 600; }
.near-btn:disabled { opacity: 0.6; cursor: default; }
.near-search { display: flex; gap: 0.3rem; flex: 1 1 12rem; min-width: 0; }
.near-search input { flex: 1; min-width: 0; font: inherit; font-size: 0.85rem; background: var(--input-bg, var(--surface)); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 0.3rem 0.5rem; }
.link-btn { font: inherit; font-size: 0.8rem; background: none; border: 0; padding: 0.2rem; color: var(--accent); cursor: pointer; text-decoration: underline; }
.dist { font-size: 0.72rem; color: var(--muted); white-space: nowrap; font-variant-numeric: tabular-nums; }
.lg i.drive { width: 14px; height: 10px; border-radius: 2px; border: 1.5px solid #7b4fd6; background: rgba(123, 79, 214, 0.12); }
.lg i.me { width: 10px; height: 10px; background: #2a78d6; border: 2px solid #fff; box-shadow: 0 0 0 1px #2a78d6; }

.filters { border: 1px solid var(--border-soft); border-radius: 8px; padding: 0.4rem 0.7rem; background: var(--surface); }
.filters > summary { cursor: pointer; display: flex; gap: 0.6rem; align-items: baseline; flex-wrap: wrap; list-style: none; }
.filters > summary::before { content: '▸'; color: var(--muted); }
.filters[open] > summary::before { content: '▾'; }
.filters > summary::-webkit-details-marker { display: none; }
.f-title { font-weight: 600; font-size: 0.9rem; }
.f-sum { font-size: 0.78rem; color: var(--muted); }
.filters[open] { display: flex; flex-direction: column; gap: 0.5rem; }
.foray-controls { margin-top: 0.4rem; display: flex; flex-wrap: wrap; gap: 0.6rem 1rem; align-items: flex-end; }
.ctl { display: flex; flex-direction: column; gap: 0.15rem; font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
.ctl input[type='date'] { font: inherit; font-size: 0.85rem; text-transform: none; letter-spacing: 0; background: var(--input-bg, var(--surface)); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 0.25rem 0.4rem; }
.ctl select { font: inherit; font-size: 0.85rem; text-transform: none; letter-spacing: 0; background: var(--input-bg, var(--surface)); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 0.3rem 0.4rem; }
.switches { display: flex; flex-wrap: wrap; gap: 0.4rem 1rem; border: 1px solid var(--border-soft); border-radius: 8px; padding: 0.4rem 0.7rem; margin: 0; font-size: 0.85rem; }
.switches:disabled { opacity: 0.5; }
.switches label { display: inline-flex; gap: 0.3rem; align-items: center; }

.notice { margin: 0; font-size: 0.78rem; color: var(--muted); }
.notice.warn { color: var(--text); background: color-mix(in srgb, #e0a800 18%, var(--surface)); border: 1px solid color-mix(in srgb, #e0a800 50%, var(--border)); border-radius: 6px; padding: 0.35rem 0.6rem; }

.foray-body { display: grid; grid-template-columns: minmax(0, 3fr) minmax(0, 2fr); gap: 0.8rem; align-items: start; }
.map-col { display: flex; flex-direction: column; gap: 0.4rem; min-width: 0; scroll-margin-top: 0.5rem; }
.map-box { height: min(55vh, 520px); }
.map-fallback { display: grid; place-items: center; height: 100%; color: var(--muted); }
.legend { display: flex; flex-wrap: wrap; align-items: center; gap: 0.3rem 0.8rem; font-size: 0.75rem; color: var(--muted); }
.lg { display: inline-flex; align-items: center; gap: 0.3rem; }
.lg i { width: 12px; height: 12px; border-radius: 50%; display: inline-block; }
.lg i.pin { width: 16px; height: 16px; font: 700 10px/16px system-ui, sans-serif; font-style: normal; text-align: center; background: #fff; color: #1b3a1f; border: 2px solid #1b5e20; }
.howto { font-size: 0.8rem; color: var(--muted); }
.howto summary { cursor: pointer; color: var(--accent); }
.howto ul { margin: 0.3rem 0 0; padding-left: 1.1rem; display: flex; flex-direction: column; gap: 0.2rem; }
.place { border: 1px solid var(--accent); border-radius: 8px; padding: 0.6rem 0.7rem; background: var(--surface-2); display: flex; flex-direction: column; gap: 0.4rem; }
.p-head { display: flex; align-items: center; gap: 0.5rem; }
.p-rank { display: grid; place-items: center; flex: 0 0 26px; height: 26px; border-radius: 50%; background: var(--accent); color: var(--accent-ink, #fff); font-weight: 700; font-size: 0.8rem; }
.p-title { flex: 1; min-width: 0; display: flex; flex-direction: column; }
.p-title strong { font-size: 0.95rem; overflow-wrap: anywhere; }
.p-sub { font-size: 0.75rem; color: var(--muted); }
.p-coords { font-size: 0.72rem; color: var(--muted); font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
.p-why { margin: 0; font-size: 0.82rem; }
.p-species { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 0.15rem; font-size: 0.82rem; }
.p-species li { display: flex; justify-content: space-between; gap: 0.5rem; }
.p-species a { color: var(--text); }
.p-n { color: var(--muted); font-size: 0.75rem; white-space: nowrap; }
.p-links { display: flex; gap: 0.5rem; flex-wrap: wrap; }
.p-links a { font-size: 0.8rem; padding: 0.3rem 0.7rem; border: 1px solid var(--border); border-radius: 6px; color: var(--text); text-decoration: none; background: var(--surface); }
.list-h { margin: 0; font-size: 0.95rem; }

.list-col { display: flex; flex-direction: column; gap: 0.6rem; min-width: 0; }
.empty { font-size: 0.85rem; color: var(--muted); padding: 0.6rem; border: 1px dashed var(--border); border-radius: 8px; }
.caveat { margin: 0; font-size: 0.75rem; color: var(--muted); border-left: 3px solid var(--border); padding-left: 0.6rem; }
.ranked { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 0.35rem; }
.ranked > li { background: var(--surface-2); border: 1px solid var(--border-soft); border-radius: 8px; padding: 0.45rem 0.6rem; cursor: pointer; }
.ranked > li:hover, .ranked > li.sel { border-color: var(--accent); }
.row1 { display: flex; align-items: center; gap: 0.5rem; }
.rk { font-variant-numeric: tabular-nums; color: var(--muted); width: 1.3rem; }
.site { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 0.9rem; }
span.site { display: flex; flex-direction: column; }
span.site strong { overflow: hidden; text-overflow: ellipsis; }
.site .coords { font-variant-numeric: tabular-nums; overflow: hidden; text-overflow: ellipsis; }
.map-link { color: var(--accent); font-size: 0.75rem; }
.xy { font-weight: 400; font-size: 0.72rem; color: var(--muted); }
.band { font-size: 0.72rem; font-weight: 700; border-radius: 999px; padding: 0.05rem 0.55rem; background: var(--surface-3); }
.band.great { background: var(--accent); color: var(--accent-ink, #fff); }
.band.good { background: color-mix(in srgb, var(--accent) 45%, var(--surface-3)); }
.row2 { font-size: 0.78rem; color: var(--muted); margin-top: 0.15rem; display: flex; flex-wrap: wrap; gap: 0.2rem 0.6rem; }
.acc { display: inline-flex; flex-wrap: wrap; gap: 0.25rem; }
.chip { font-size: 0.68rem; border: 1px solid var(--border); border-radius: 4px; padding: 0 0.3rem; color: var(--text); }
.chip.ok { border-color: var(--accent); }
.comp { font-size: 0.72rem; color: var(--muted); margin-top: 0.3rem; }
.comp ul { margin: 0.2rem 0 0; padding-left: 1rem; }

.chk { display: inline-flex; gap: 0.3rem; align-items: center; font-size: 0.85rem; min-height: 32px; }
.tune { margin-top: 0.4rem; display: flex; flex-direction: column; gap: 0.55rem; }
.tune-row { display: flex; flex-wrap: wrap; align-items: center; gap: 0.4rem 0.8rem; font-size: 0.85rem; }
.tune-l { color: var(--muted); }
.tune-s { display: flex; flex-direction: column; gap: 0.2rem; font-size: 0.82rem; }
.tune-s input[type='range'] { width: 100%; max-width: 26rem; min-height: 28px; }
.tune-actions { display: flex; gap: 0.4rem; flex-wrap: wrap; }
.grad { display: flex; flex-direction: column; gap: 0.15rem; font-size: 0.75rem; color: var(--muted); }
.grad-bar { position: relative; height: 12px; border-radius: 6px; border: 1px solid var(--border-soft); }
.grad-tick { position: absolute; top: -2px; bottom: -2px; width: 1px; background: var(--text); opacity: 0.45; }
.grad-sel { position: absolute; top: 12px; width: 0; height: 0; margin-left: -6px; border: 6px solid transparent; border-bottom-color: var(--text); border-top: 0; }
.grad-labels { display: grid; grid-template-columns: repeat(4, 1fr); text-align: center; margin-top: 4px; color: var(--text); }
.grad-cap { font-size: 0.7rem; }
.lg i.pred { border: 2px dashed #41ab5d; border-radius: 2px; background: transparent; }
.chip.pred-chip { border-style: dashed; border-color: #9c4dcc; color: color-mix(in srgb, #9c4dcc 70%, var(--text)); white-space: nowrap; }
.why { font-size: 0.8rem; border-top: 1px solid var(--border-soft); padding-top: 0.35rem; }
.why > summary { cursor: pointer; color: var(--accent); }
.why-steps { margin: 0.35rem 0; padding-left: 1.1rem; display: flex; flex-direction: column; gap: 0.25rem; }
.why-table { margin-top: 0.2rem; font-size: 0.74rem; }
.why-table td, .why-table th { padding: 0.15rem 0.25rem; }
@media (max-width: 480px) { .why-table th:nth-child(3), .why-table td:nth-child(3) { display: none; } }

.shortlist, .under { border: 1px solid var(--border); border-radius: 8px; padding: 0.6rem; background: var(--surface); display: flex; flex-direction: column; gap: 0.4rem; }
.rk.opp { color: #6a1b9a; font-weight: 700; }
.lg i.pin.opp { color: #4a148c; border-color: #6a1b9a; border-style: dashed; }
.lg i.ramp { width: 36px; height: 10px; border-radius: 3px; }
.model-list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 0.25rem; font-size: 0.85rem; }
.model-list label { display: flex; flex-wrap: wrap; align-items: center; gap: 0.2rem 0.5rem; min-height: 32px; }
.model-list input { width: 18px; height: 18px; }
.m-title { font-weight: 600; }
.m-meta { font-size: 0.75rem; color: var(--muted); }
.ctl input[type='range'] { width: 8rem; }
.sl-head { display: flex; justify-content: space-between; align-items: center; gap: 0.5rem; flex-wrap: wrap; }
.sl-head h3 { margin: 0; font-size: 0.95rem; }
.sl-actions { display: flex; gap: 0.4rem; }
.sl-actions > button { font: inherit; font-size: 0.78rem; padding: 0.25rem 0.6rem; border: 1px solid var(--border); border-radius: 6px; background: var(--surface-2); color: var(--text); cursor: pointer; }
.sl-actions > button:disabled { opacity: 0.5; cursor: default; }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 0.78rem; margin-top: 0.5rem; }
th, td { text-align: left; padding: 0.25rem 0.3rem; border-bottom: 1px solid var(--border-soft); vertical-align: top; }
.coords { color: var(--muted); font-size: 0.68rem; }
.note-in { width: 100%; min-width: 7rem; font: inherit; font-size: 0.75rem; background: var(--input-bg, var(--surface)); color: var(--text); border: 1px solid var(--border); border-radius: 4px; padding: 0.15rem 0.3rem; }
.note-print, .print-disclaimer { display: none; }

@media (max-width: 800px) {
  .foray-body { grid-template-columns: 1fr; }
  .map-box { height: 52vh; }
}
</style>

<style>
@media print {
  body:has(.foray-page) .app-header,
  body:has(.foray-page) .foray-head,
  body:has(.foray-page) .filters,
  body:has(.foray-page) .near,
  body:has(.foray-page) .list-h,
  body:has(.foray-page) .notice,
  body:has(.foray-page) .map-col,
  body:has(.foray-page) .ranked,
  body:has(.foray-page) .under,
  body:has(.foray-page) .caveat,
  body:has(.foray-page) .empty,
  body:has(.foray-page) .sl-actions,
  body:has(.foray-page) .note-in { display: none !important; }
  body:has(.foray-page) .foray-page { height: auto; overflow: visible; color: #000; }
  body:has(.foray-page) .foray-body { display: block; }
  body:has(.foray-page) .note-print, body:has(.foray-page) .print-disclaimer { display: block; }
  body:has(.foray-page) .shortlist { border: 0; }
}
</style>
