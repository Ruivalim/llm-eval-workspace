import { readSse } from './sse'

export type LocaleText = { name: string; system_prompt: string; prompt: string }
export type EvalType = 'text' | 'render_html' | 'tool_call' | 'error_test'
export type Prompt = {
  id: string
  category: string
  eval_type: EvalType
  max_tokens: number | null
  tags: string[]
  tools: string[]
  fault: Record<string, unknown> | null
  check: { type: string; params: Record<string, unknown> }
  locales: Record<string, LocaleText>
  auto: boolean
}
export type Category = { slug: string; icon: string; labels: Record<string, string> }
export type Meta = { locales: string[]; default_locale: string; categories: Category[]; prompts: Prompt[] }
export type Server = { id: number; name: string; flavor: Flavor; base_url: string; has_key: boolean }
export type Flavor = 'openai' | 'unsloth' | 'anthropic'
export type Model = { id: string; loaded: boolean }
/** Thinking effort. default sends nothing; off turns thinking off; the rest is a level. */
export const EFFORTS = ['default', 'off', 'low', 'medium', 'high', 'xhigh', 'max'] as const
export type Effort = (typeof EFFORTS)[number]
/** Where a cost came from: the endpoint billed it, your price, or the built-in list price. */
export type CostSource = 'server' | 'price' | 'builtin' | ''
export type Cost = { cost_usd: number | null; cost_source: CostSource }
export type ModelPrice = { model: string; input_per_mtok: number; output_per_mtok: number }
export type Prices = { prices: ModelPrice[]; builtin: ModelPrice[]; builtin_as_of: string }
export type Telemetry = {
  ttft_ms: number
  total_ms: number
  tokens_in: number
  tokens_out: number
  tokens_think: number
  tps: number
  usage_from_server: boolean
}
export type Status = 'PASS' | 'FAIL' | 'ERROR' | 'PARTIAL' | 'PENDING'
export type Outcome = {
  result_id: number | null
  status: Status
  detail: string
  error: string | null
  artifact: string | null
  trace: boolean
  finish_reason: string | null
  answer: string
  reasoning: string
  tool_calls: string[]
  telemetry: Telemetry | null
} & Cost
export type DeltaKind = 'content' | 'reasoning' | 'tool_call' | 'tool_result'
export type Delta = { kind: DeltaKind; text: string }
export type HistoryRow = {
  id: number
  run_id: number
  prompt_id: string
  eval_type: EvalType
  status: Status
  detail: string
  artifact: string | null
  trace: boolean
  temperature: number | null
  max_tokens: number | null
  ttft_ms: number
  total_ms: number
  tokens_in: number
  tokens_out: number
  tokens_think: number
  tps: number
  created_at: string
  model: string
  locale: string
  server_id: number
  server_name: string
  /** Requests the batch had in flight when this ran. 1 = sequential. */
  parallel: number
  effort: Effort
  human_score: number | null
} & Cost
export type ResultRow = HistoryRow & { answer: string; reasoning: string; human_comment: string | null }
export type LeaderboardRow = {
  model: string
  effort: Effort
  runs: number
  passed: number
  prompts: number
  pass_rate: number
  avg_ttft_ms: number
  avg_tps: number
  avg_human: number | null
  /** Sum over results with a known cost; `priced` says how many that was. */
  total_cost: number | null
  priced: number
}
export type MatrixCell = {
  id: number
  prompt_id: string
  status: Status
  tps: number
  ttft_ms: number
  detail: string
  model: string
  effort: Effort
  locale: string
  server_id: number
  human_score: number | null
}
export type EvaluateRequest = {
  server_id: number
  model: string
  prompt_id: string
  locale: string
  temperature: number
  top_p: number
  timeout_s: number
  effort: Effort
  system_override: string | null
  parallel: number
}

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

function query<T>(path: string, q: Record<string, string> = {}): Promise<T> {
  const params = new URLSearchParams()
  for (const [key, value] of Object.entries(q)) if (value) params.set(key, value)
  return request(`${path}?${params}`)
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, { ...init, headers: { 'content-type': 'application/json', ...(init?.headers ?? {}) } })
  if (!r.ok) {
    let msg = r.statusText
    try {
      const j = await r.json()
      msg = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail ?? j)
    } catch {
      /* keep statusText */
    }
    throw new ApiError(r.status, msg)
  }
  if (r.status === 204) return undefined as T
  return (await r.json()) as T
}

export type AppSettings = { global_system_prompt: string }

export const api = {
  meta: () => request<Meta>('/api/meta'),
  settings: () => request<AppSettings>('/api/settings'),
  saveSettings: (body: AppSettings) => request<AppSettings>('/api/settings', { method: 'PUT', body: JSON.stringify(body) }),
  servers: () => request<Server[]>('/api/servers'),
  addServer: (body: { name: string; base_url: string; api_key: string | null; flavor: string }) =>
    request<Server>('/api/servers', { method: 'POST', body: JSON.stringify(body) }),
  updateServer: (id: number, body: { name: string; base_url: string; flavor: string; api_key?: string }) =>
    request<Server>(`/api/servers/${id}`, { method: 'PUT', body: JSON.stringify(body) }),
  deleteServer: (id: number) => request<void>(`/api/servers/${id}`, { method: 'DELETE' }),
  models: (serverId: number) => request<Model[]>(`/api/servers/${serverId}/models`),
  loadModel: (serverId: number, model: string) =>
    request<{ loaded: string }>(`/api/servers/${serverId}/load`, { method: 'POST', body: JSON.stringify({ model }) }),
  prices: () => request<Prices>('/api/prices'),
  setPrice: (body: ModelPrice) => request<{ model: string; filled: number }>('/api/prices', { method: 'PUT', body: JSON.stringify(body) }),
  deletePrice: (model: string) => request<void>(`/api/prices?${new URLSearchParams({ model })}`, { method: 'DELETE' }),
  history: (q: Partial<Record<'model' | 'prompt_id' | 'status' | 'locale', string>> & { limit?: number } = {}) => {
    const qs = new URLSearchParams()
    for (const [k, v] of Object.entries(q)) if (v !== undefined && v !== '') qs.set(k, String(v))
    return request<HistoryRow[]>(`/api/history?${qs}`)
  },
  leaderboard: (q: Partial<Record<'model' | 'prompt_id' | 'locale', string>> = {}) => query<LeaderboardRow[]>('/api/leaderboard', q),
  matrix: (q: Partial<Record<'model' | 'prompt_id' | 'locale', string>> = {}) => query<MatrixCell[]>('/api/matrix', q),
  result: (id: number) => request<ResultRow>(`/api/results/${id}`),
  score: (id: number, score: number, comment: string) =>
    request<void>(`/api/results/${id}/score`, { method: 'POST', body: JSON.stringify({ score, comment }) }),
  deleteResult: (id: number) => request<void>(`/api/results/${id}`, { method: 'DELETE' }),
  clearResults: (q: Partial<Record<'model' | 'prompt_id' | 'status' | 'locale', string>> = {}) => {
    const qs = new URLSearchParams()
    for (const [k, v] of Object.entries(q)) if (v) qs.set(k, String(v))
    return request<{ deleted: number }>(`/api/results?${qs}`, { method: 'DELETE' })
  },

  /** Stream one evaluation. Resolves with the outcome; deltas go to onDelta. */
  async evaluate(body: EvaluateRequest, onDelta: (d: Delta) => void, signal?: AbortSignal): Promise<Outcome> {
    const r = await fetch('/api/evaluate', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
      signal,
    })
    if (!r.ok || !r.body) {
      let msg = r.statusText
      try {
        msg = (await r.json()).detail ?? msg
      } catch {
        /* keep */
      }
      throw new ApiError(r.status, typeof msg === 'string' ? msg : JSON.stringify(msg))
    }
    for await (const ev of readSse(r.body)) {
      if (ev.event === 'delta') onDelta(JSON.parse(ev.data) as Delta)
      else if (ev.event === 'outcome') return JSON.parse(ev.data) as Outcome
      else if (ev.event === 'error') throw new ApiError(500, (JSON.parse(ev.data) as { message: string }).message)
    }
    throw new ApiError(500, 'stream ended without an outcome')
  },
}

export function traceUrl(resultId: number, raw = false): string {
  return `/api/results/${resultId}/trace${raw ? '?raw=1' : ''}`
}

export function artifactUrl(name: string): string {
  return `/api/artifacts/${encodeURIComponent(name)}`
}
