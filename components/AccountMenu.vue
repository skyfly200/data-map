<template>
  <div ref="root" class="acct">
    <button class="acct-btn" :class="{ on: open }" :aria-expanded="String(open)"
            :title="user?.email || 'Account and display settings'" @click="open = !open">
      <!-- Signed in, the initial is the badge and it is the same at every
           width. Signed out, a bare gear said nothing about there being an
           account to make: it now says so in words, which fit because they
           replace the email that is not there. -->
      <span class="avatar" :class="{ anon: !signedIn }">{{ signedIn ? initial : '☺' }}</span>
      <span v-if="signedIn" class="who">{{ shortEmail }}</span>
      <span v-else-if="configured" class="who signin">Sign in</span>
      <!-- The sync dot rides on the button so its state is visible with the
           menu shut. Burying a failure inside a menu nobody opens is how a
           broken sync goes unnoticed for a week. -->
      <span v-if="syncState" class="mini-dot" :class="syncState" :title="syncTitle"></span>
      <span class="caret" aria-hidden="true">▾</span>
    </button>

    <div v-if="open" class="acct-menu">
      <div v-if="signedIn" class="acct-head">
        <span class="avatar big">{{ initial }}</span>
        <span class="email" :title="user?.email || ''">{{ user?.email }}</span>
      </div>

      <div v-if="signedIn" class="acct-sync"><SyncStatus /></div>

      <!-- Units and theme live here rather than in the header. They are set once
           and then left alone for months, and they were taking two thirds of the
           header's width on a phone to do it.

           Which is also why this menu is shown signed out: it is no longer only
           an account menu, and someone who has not made an account still needs
           to switch to metric. -->
      <div class="acct-prefs">
        <div class="pref">
          <span class="pref-label">Elevation</span>
          <div class="seg" role="group" aria-label="Elevation units">
            <button :class="{ active: unit === 'ft' }" @click="unit = 'ft'">ft</button>
            <button :class="{ active: unit === 'm' }" @click="unit = 'm'">m</button>
          </div>
        </div>
        <div class="pref">
          <span class="pref-label">Temperature</span>
          <div class="seg" role="group" aria-label="Temperature units">
            <button :class="{ active: tempUnit === 'F' }" @click="tempUnit = 'F'">°F</button>
            <button :class="{ active: tempUnit === 'C' }" @click="tempUnit = 'C'">°C</button>
          </div>
        </div>
        <div class="pref">
          <span class="pref-label">Theme</span>
          <div class="seg" role="group" aria-label="Theme">
            <button :class="{ active: theme === 'dark' }" @click="$emit('set-theme', 'dark')">☾</button>
            <button :class="{ active: theme === 'light' }" @click="$emit('set-theme', 'light')">☀</button>
          </div>
        </div>
      </div>

      <!-- Members get the pipeline; admins also get the console. Hidden rather
           than disabled for people whose tier does not reach them, since a menu
           of things you cannot do is not useful. The server gates both anyway;
           this is only about what is worth showing. -->
      <NuxtLink v-if="isMember" to="/jobs" class="acct-item" @click="open = false">Pipeline jobs</NuxtLink>
      <NuxtLink v-if="isAdmin" to="/admin" class="acct-item" @click="open = false">Administration</NuxtLink>
      <NuxtLink to="/options" class="acct-item" @click="open = false">⚙ Options</NuxtLink>
      <button v-if="signedIn" class="acct-item danger" @click="onSignOut">Sign out</button>
      <NuxtLink v-else-if="configured" to="/login" class="acct-item" @click="open = false">Sign in</NuxtLink>
    </div>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useUnits } from '~/composables/useUnits'

const props = defineProps({
  signedIn: { type: Boolean, default: false },
  configured: { type: Boolean, default: false },
  theme: { type: String, default: 'dark' },
  user: { type: Object, default: null },
  initial: { type: String, default: '?' },
  shortEmail: { type: String, default: '' },
})
const emit = defineEmits(['sign-out', 'set-theme'])

const { unit, tempUnit } = useUnits()

// Read from the token's claim, so the menu is right on first paint without a
// request of its own.
const { isMember, isAdmin } = useMembership()

const cloud = safeCloudSync()
const syncState = computed(() => {
  if (!cloud?.enabled?.value && cloud?.status?.value !== 'error') return ''
  return cloud?.status?.value || ''
})
const syncTitle = computed(() => (syncState.value === 'error'
  ? 'Sync failed, open the menu for details'
  : `Sync: ${syncState.value}`))

const open = ref(false)
const root = ref(null)

function onSignOut() {
  open.value = false
  emit('sign-out')
}

function onDocClick(e) {
  if (open.value && root.value && !root.value.contains(e.target)) open.value = false
}
onMounted(() => document.addEventListener('click', onDocClick))
onBeforeUnmount(() => document.removeEventListener('click', onDocClick))
</script>

<style scoped>
.acct { position: relative; }

.acct-btn {
  display: inline-flex; align-items: center; gap: 7px;
  border: 1px solid #52606d; background: transparent; color: #cbd2d9;
  border-radius: 6px; padding: 4px 8px; font-size: 0.82rem; font-weight: 600; cursor: pointer;
}
.acct-btn:hover, .acct-btn.on { background: rgba(255, 255, 255, 0.08); color: #fff; }
.avatar {
  display: inline-flex; align-items: center; justify-content: center;
  width: 22px; height: 22px; border-radius: 50%; background: #3e4c59; color: #fff;
  font-size: 0.7rem; font-weight: 700; flex: 0 0 auto;
}
.avatar.big { width: 30px; height: 30px; font-size: 0.85rem; }
.who { max-width: 130px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.caret { font-size: 9px; opacity: 0.8; }
.mini-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--muted); flex: 0 0 auto; }
.mini-dot.ok, .mini-dot.idle { background: #2b7a3d; }
.mini-dot.syncing { background: #eda100; }
.mini-dot.error { background: #e34948; }

.acct-menu {
  position: absolute; top: calc(100% + 6px); right: 0; z-index: 900; width: 240px;
  background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  box-shadow: 0 4px 16px var(--shadow); padding: 8px; color: var(--text);
}
.acct-head {
  display: flex; align-items: center; gap: 8px; padding: 4px 6px 10px;
  border-bottom: 1px solid var(--border-soft, var(--border)); margin-bottom: 6px;
}
.acct-head .email {
  font-size: 0.8rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.acct-sync { padding: 2px 6px 8px; }

.acct-item {
  display: block; width: 100%; text-align: left; border: 0; background: transparent;
  color: var(--text); border-radius: 6px; padding: 7px 8px; font-size: 0.85rem;
  cursor: pointer; text-decoration: none;
}
.acct-item:hover { background: var(--surface-2); }
.acct-prefs {
  display: flex; flex-direction: column; gap: 7px;
  padding: 8px 4px; margin-bottom: 4px;
  border-bottom: 1px solid var(--border);
}
.pref { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.pref-label { font-size: 0.8rem; color: var(--muted); }
.seg { display: inline-flex; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
.seg button {
  background: transparent; border: 0; color: var(--muted);
  padding: 4px 10px; font: inherit; font-size: 0.78rem; cursor: pointer; min-width: 38px;
}
.seg button:hover { background: var(--surface-2, rgba(127, 127, 127, 0.12)); color: var(--text); }
.seg button.active { background: var(--accent, #2b7a3d); color: #fff; }
.avatar.anon { background: transparent; border: 1px solid #52606d; color: #cbd2d9; }
/* The email is the first thing worth losing when space is short, because the
   avatar already says who is signed in. "Sign in" is the opposite: it IS the
   message, so it stays at every width. */
.who.signin { display: inline !important; font-weight: 600; }

.acct-item.danger { color: var(--danger, #b00020); }

@media (max-width: 480px) {
  .who { display: none; }
  .acct-menu { width: min(240px, calc(100vw - 24px)); }
}
</style>
