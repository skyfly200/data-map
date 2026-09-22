// Lightweight in-app alert banner for surfacing errors that would otherwise
// only appear in the browser console. Components show these with <AppAlerts />.
//
// An alert auto-dismisses after `ttl` ms (default 8 s). Errors stay until
// dismissed by the user. Info and warning levels auto-dismiss.

export type AlertLevel = 'error' | 'warn' | 'info'

export interface AppAlert {
  id: number
  level: AlertLevel
  message: string
  ttl?: number
}

let nextId = 1

const alerts = useState<AppAlert[]>('app-alerts', () => [])

export function useAppAlerts() {
  function push(level: AlertLevel, message: string, ttl?: number) {
    const id = nextId++
    const defaultTtl = level === 'error' ? undefined : 8000
    alerts.value = [...alerts.value, { id, level, message, ttl: ttl ?? defaultTtl }]
    if (ttl !== undefined || level !== 'error') {
      const delay = ttl ?? defaultTtl ?? 8000
      if (import.meta.client) setTimeout(() => dismiss(id), delay)
    }
    return id
  }

  function dismiss(id: number) {
    alerts.value = alerts.value.filter((a) => a.id !== id)
  }

  function error(message: string) { return push('error', message) }
  function warn(message: string, ttl = 8000) { return push('warn', message, ttl) }
  function info(message: string, ttl = 6000) { return push('info', message, ttl) }

  return { alerts, push, dismiss, error, warn, info }
}
