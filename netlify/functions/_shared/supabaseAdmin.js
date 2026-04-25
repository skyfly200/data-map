import { createClient } from '@supabase/supabase-js'

// Service-role client — bypasses RLS. Only used in server-side functions.
// Never expose SUPABASE_SERVICE_ROLE_KEY to the frontend.
export const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
)
