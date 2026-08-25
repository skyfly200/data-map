<template>
  <div class="legal">
    <article>
      <h1>Privacy Policy</h1>
      <p class="updated">Last updated: {{ updated }}</p>

      <p>
        This Privacy Policy explains what information this application
        (“the app”) collects, why, and how it is handled. The app maps public
        iNaturalist mushroom observations enriched with environmental data. It
        is a small, non-commercial project.
      </p>

      <h2>Information we collect</h2>
      <ul>
        <li>
          <strong>Account information.</strong> If you choose to sign in, our
          authentication provider (Supabase Auth) stores the identifier you sign
          in with — an email address, or the basic profile identifier returned
          by an OAuth provider (GitHub or Google) or a passkey credential. We use
          this only to recognise your account and to rate-limit the data-fetching
          features so they aren’t abused.
        </li>
        <li>
          <strong>Browser storage.</strong> The app stores small preferences in
          your browser (chosen units, selected dataset, saved charts) and a
          session token so you stay signed in. This stays in your browser.
        </li>
        <li>
          <strong>Observation data.</strong> The mushroom observations shown are
          public records retrieved from iNaturalist. They are not your personal
          data.
        </li>
      </ul>
      <p>
        The app does not collect payment information and does not run
        advertising or third-party tracking.
      </p>

      <h2>How we use information</h2>
      <ul>
        <li>To authenticate you and keep you signed in.</li>
        <li>To limit access to the on-demand data-fetching features to signed-in users.</li>
        <li>To remember your display preferences.</li>
      </ul>

      <h2>Third-party services</h2>
      <p>The app relies on these services, each with its own privacy policy:</p>
      <ul>
        <li><strong>Supabase</strong> — authentication and data storage.</li>
        <li><strong>Netlify</strong> — hosting and serverless functions.</li>
        <li><strong>iNaturalist</strong> — source of the public observation data.</li>
        <li><strong>GitHub / Google</strong> — optional OAuth sign-in providers, used only if you choose them.</li>
      </ul>

      <h2>Data sharing</h2>
      <p>
        We do not sell your information and do not share it except as needed to
        operate the services listed above.
      </p>

      <h2>Data retention and deletion</h2>
      <p>
        Account information is retained while your account exists. To delete your
        account and associated authentication data, contact us at the address
        below. Preferences stored in your browser can be cleared at any time by
        clearing your browser’s site data.
      </p>

      <h2>Contact</h2>
      <p>
        Questions about this policy can be sent to
        <a href="mailto:CONTACT_EMAIL_HERE">CONTACT_EMAIL_HERE</a>.
        <em class="note">(Replace this with your contact email before publishing.)</em>
      </p>
    </article>
  </div>
</template>

<script setup>
useHead({ title: 'Privacy Policy · data-map' })
// Static “last updated” date; bump when you revise the policy.
const updated = 'August 25, 2026'
</script>

<style scoped>
.legal { max-width: 760px; margin: 0 auto; padding: 32px 20px 64px; }
article { color: #1f2933; line-height: 1.55; }
h1 { font-size: 1.5rem; margin: 0 0 4px; }
.updated { color: #6b7280; font-size: 0.85rem; margin: 0 0 20px; }
h2 { font-size: 1.05rem; margin: 26px 0 8px; }
p { margin: 0 0 12px; }
ul { margin: 0 0 12px; padding-left: 20px; }
li { margin: 4px 0; }
a { color: #2b7a3d; }
.note { color: #9aa0a6; font-style: italic; font-size: 0.85rem; }
</style>
