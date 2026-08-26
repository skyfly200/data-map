<template>
  <div class="home">
    <section class="hero">
      <h1>Mushroom Observations, mapped and enriched</h1>
      <p class="lead">
        iNaturalist mushroom finds enriched with terrain and environmental exposure —
        elevation, solar and wind exposure, water retention, land cover, and weather —
        then clustered by similarity. Explore them on an interactive map, in tables, and
        through charts you can build yourself.
      </p>

      <div class="cta">
        <NuxtLink to="/map" class="btn primary">Open the map</NuxtLink>
        <NuxtLink to="/explore" class="btn">Build a chart</NuxtLink>
      </div>

      <ClientOnly>
        <div class="auth-cta" v-if="configured">
          <template v-if="isAuthed">
            <span class="signed">Signed in as <strong>{{ user?.email || 'your account' }}</strong>.</span>
            <NuxtLink to="/data" class="btn small">Fetch a species</NuxtLink>
            <button class="btn small ghost" @click="signOut">Sign out</button>
          </template>
          <template v-else>
            <span class="hint">Sign in to fetch new species on demand:</span>
            <NuxtLink to="/login" class="btn small">Sign in</NuxtLink>
            <NuxtLink to="/login?mode=signup" class="btn small ghost">Sign up</NuxtLink>
          </template>
        </div>
      </ClientOnly>

      <a class="repo" :href="repoUrl" target="_blank" rel="noopener noreferrer">
        <IconGithub class="repo-ico" /> <span>View the source on GitHub</span>
      </a>
    </section>

    <section class="features">
      <div class="feature">
        <h3>Interactive map</h3>
        <p>Every observation as a point, coloured by environmental cluster, with the enriched attributes in each popup.</p>
      </div>
      <div class="feature">
        <h3>Filter by place &amp; time</h3>
        <p>Narrow by country, state, county or a radius, and by year, month, week, or date range — across every view at once.</p>
      </div>
      <div class="feature">
        <h3>Charts you build</h3>
        <p>Scatter, bar, box, histogram and heatmap. Compose your own on the Explore tab and save the ones you like.</p>
      </div>
    </section>
  </div>
</template>

<script setup>
const { user, isAuthed, configured, signOut } = useAuth()
const repoUrl = 'https://github.com/skyfly200/data-map'

useHead({ title: 'data-map · Mushroom Observations' })
</script>

<style scoped>
.home { max-width: 900px; margin: 0 auto; padding: 40px 20px 64px; }

.hero { text-align: center; }
.hero h1 { font-size: 1.9rem; margin: 0 0 14px; line-height: 1.2; color: var(--text); }
.lead { max-width: 640px; margin: 0 auto 24px; color: var(--text); font-size: 1rem; line-height: 1.6; }

.cta { display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; margin-bottom: 20px; }
.btn {
  display: inline-flex; align-items: center; gap: 8px; text-decoration: none; cursor: pointer;
  border: 1px solid var(--border); background: var(--surface); color: var(--text); border-radius: 8px;
  padding: 10px 18px; font-size: 0.92rem; font-weight: 600;
}
.btn:hover { background: var(--surface-2); }
.btn.primary { background: #2b7a3d; border-color: #2b7a3d; color: #fff; }
.btn.primary:hover { background: #246833; }
.btn.small { padding: 6px 12px; font-size: 0.82rem; }
.btn.ghost { background: var(--surface); }

.auth-cta {
  display: inline-flex; align-items: center; gap: 10px; flex-wrap: wrap; justify-content: center;
  background: var(--surface-2); border: 1px solid var(--border); border-radius: 10px; padding: 10px 16px; margin-bottom: 22px;
}
.auth-cta .hint, .auth-cta .signed { font-size: 0.85rem; color: var(--muted); }

.repo { display: inline-flex; align-items: center; gap: 8px; color: var(--text); text-decoration: none; font-size: 0.88rem; font-weight: 600; }
.repo:hover { color: var(--text); text-decoration: underline; }
.repo-ico { width: 18px; height: 18px; }

.features { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-top: 48px; }
.feature { border: 1px solid var(--border); border-radius: 10px; padding: 16px 18px; background: var(--surface); }
.feature h3 { margin: 0 0 6px; font-size: 1rem; color: var(--text); }
.feature p { margin: 0; font-size: 0.86rem; color: var(--muted); line-height: 1.5; }
</style>
