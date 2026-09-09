// What the signed-in viewer's membership entitles them to, for the UI.
//
// The tier is read out of the access token, where the Supabase custom access
// token hook stamped it (see supabase_migrations/002_membership_jobs_and_admin
// .sql). That means the header can show the right thing immediately, with no
// request of its own, and it stays right across a reload because the session
// is already persisted.
//
// This is for SHOWING things, never for allowing them. Everything here runs in
// the viewer's browser, where it can be edited; the gate that matters is
// requireTier / requireMemberFresh in netlify/lib/auth.mjs. Hiding the admin
// link is a courtesy to people who cannot use it, not a security control.
//
// The tier vocabulary is imported from the server module on purpose: two copies
// of "what counts as a member" would eventually disagree, and the direction
// they would disagree in is the expensive one.

import { atLeast, TIER_LABELS, tierFromToken } from '../netlify/lib/tiers.mjs'

export function useMembership() {
  const { $supabase } = useNuxtApp()
  const { user, ready, configured, isAuthed } = useAuth()

  // Kept in useState so the header, the jobs page and the admin console agree
  // without each decoding the token for itself.
  const tier = useState('membership-tier', () => 'free')
  const profile = useState('membership-profile', () => null)
  const profileError = useState('membership-profile-error', () => '')

  /** Re-read the tier from the current session's token. */
  async function refresh() {
    if (!configured || !$supabase) {
      // No Supabase means no accounts at all, and the app is fully open. Saying
      // 'free' would hide features from someone who has no way to sign in.
      tier.value = 'admin'
      return tier.value
    }
    const { data } = await $supabase.auth.getSession()
    const token = data.session?.access_token
    tier.value = token ? tierFromToken(token) : 'free'
    return tier.value
  }

  /**
   * The profile row: quotas, limits, and when membership lapses.
   *
   * Separate from the tier because it costs a request and most screens do not
   * need it. RLS returns only the viewer's own row.
   */
  async function loadProfile() {
    if (!configured || !$supabase || !user.value) { profile.value = null; return null }
    profileError.value = ''
    const { data, error } = await $supabase
      .from('profiles')
      .select('tier, member_until, ee_quota_monthly, ee_jobs_per_day, ee_max_points, ee_max_concurrent, display_name')
      .eq('user_id', user.value.id)
      .maybeSingle()
    if (error) {
      // The usual cause is migration 002 not having been run yet, which is
      // worth saying plainly rather than leaving the panel blank.
      profileError.value = /schema cache|does not exist/i.test(error.message || '')
        ? 'Membership tables are missing. Run supabase_migrations/002_membership_jobs_and_admin.sql.'
        : 'Could not read your membership.'
      profile.value = null
      return null
    }
    profile.value = data || null
    return profile.value
  }

  if (import.meta.client) {
    // The token changes on sign-in, sign-out and every refresh, and the tier
    // rides along with it, so this is the one place that has to notice.
    watch([() => user.value?.id, ready], () => { refresh() }, { immediate: true })
  }

  const isMember = computed(() => atLeast(tier.value, 'member'))
  const isAdmin = computed(() => atLeast(tier.value, 'admin'))
  const label = computed(() => TIER_LABELS[tier.value] || TIER_LABELS.free)

  /**
   * Membership that is real but not yet visible.
   *
   * A tier change only reaches the browser when the token next refreshes, so
   * someone who has just been upgraded keeps seeing the old answer for up to an
   * hour. Rather than let that look like a bug, the UI can offer this: it forces
   * a refresh and re-reads the claim.
   */
  async function refreshSession() {
    if (!configured || !$supabase) return tier.value
    await $supabase.auth.refreshSession()
    return refresh()
  }

  const lapsesAt = computed(() => (profile.value?.member_until
    ? new Date(profile.value.member_until)
    : null))

  return {
    tier, label, isMember, isAdmin, isAuthed, ready, configured,
    profile, profileError, lapsesAt,
    refresh, refreshSession, loadProfile,
  }
}
