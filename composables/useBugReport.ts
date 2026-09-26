const REPO = 'skyfly200/data-map'
const MAX_LOG_LINES = 50

// Shared rolling buffer — populated by startLogCapture(), read by reportBug().
const logBuffer: string[] = []

function appendLog(line: string) {
  logBuffer.push(line)
  if (logBuffer.length > MAX_LOG_LINES) logBuffer.shift()
}

let capturing = false

export function startLogCapture() {
  if (!import.meta.client || capturing) return
  capturing = true

  const _error = console.error.bind(console)
  const _warn  = console.warn.bind(console)

  console.error = (...args: unknown[]) => {
    appendLog(`[error] ${args.map(String).join(' ')}`)
    _error(...args)
  }
  console.warn = (...args: unknown[]) => {
    appendLog(`[warn] ${args.map(String).join(' ')}`)
    _warn(...args)
  }

  window.addEventListener('error', (e) => {
    appendLog(`[uncaught] ${e.message} @ ${e.filename}:${e.lineno}`)
  })
  window.addEventListener('unhandledrejection', (e) => {
    appendLog(`[promise] ${String(e.reason)}`)
  })
}

export function useBugReport() {
  function reportBug(message: string) {
    const logs = logBuffer.slice().join('\n') || '(no logs captured)'
    const userAgent = import.meta.client ? navigator.userAgent : ''
    const url = import.meta.client ? location.href : ''

    const body = [
      `## Error\n${message}`,
      `## Recent Logs\n\`\`\`\n${logs}\n\`\`\``,
      `## Context\n- URL: ${url}\n- UA: ${userAgent}`,
    ].join('\n\n')

    const issueUrl = `https://github.com/${REPO}/issues/new?`
      + `title=${encodeURIComponent(`Bug: ${message.slice(0, 80)}`)}`
      + `&body=${encodeURIComponent(body)}`
      + `&labels=bug`

    window.open(issueUrl, '_blank', 'noopener')
  }

  return { reportBug }
}
