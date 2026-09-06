export default defineEventHandler((event) => {
  setHeader(event, 'Content-Type', 'application/json')
  setHeader(event, 'Cache-Control', 'public, max-age=0, must-revalidate')
  return {
    id: '',
    timestamp: 0,
    matcher: { static: {}, wildcard: {}, dynamic: {} },
    prerendered: [],
  }
})
