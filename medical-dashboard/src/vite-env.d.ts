/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL: string
  readonly VITE_SSE_ENDPOINT: string
  readonly VITE_HEALTH_CHECK_INTERVAL: string
  readonly VITE_MAX_DEVICES: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
