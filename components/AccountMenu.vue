<template>
  <div ref="root" class="acct">
    <button class="acct-btn" :class="{ on: open }" :aria-expanded="String(open)"
            :title="user?.email || 'Account'" @click="open = !open">
      <span class="avatar">{{ initial }}</span>
      <span class="who">{{ shortEmail }}</span>
      <!-- The sync dot rides on the button so its state is visible with the
           menu shut. Burying a failure inside a menu nobody opens is how a
           broken sync goes unnoticed for a week. -->
      <span v-if="syncState" class="mini-dot" :class="syncState" :title="syncTitle"></span>
      <span class="caret" aria-hidden="true">▾</span>
    </button>

    <div v-if="open" class="acct-menu">
      <div class="acct-head">
        <span class="avatar big">{{ initial }}</span>
        <span class="email" :title="user?.email || ''">{{ user?.email || 'Signed in' }}</span>
      </div>

      <div class="acct-sync"><SyncStatus /></div>

      <NuxtLink to="/options" class="acct-item" @click="open = false">⚙ Options</NuxtLink>
      <NuxtLink to="/guide" class="acct-item" @click="open = false">Guide</NuxtLink>
      <button class="acct-item danger" @click="onSignOut">Sign out</button>
    </div>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'

const props = defineProps({
  user: { type: Object, default: null },
  initial: { type: String, default: '?' },
  shortEmail: { type: String, default: '' },
})
const emit = defineEmits(['sign-out'])

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
.acct-item.danger { color: var(--danger, #b00020); }

@media (max-width: 480px) {
  .who { display: none; }
  .acct-menu { width: min(240px, calc(100vw - 24px)); }
}
</style>
