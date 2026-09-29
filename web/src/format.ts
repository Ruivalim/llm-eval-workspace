export const ms = (v: number | null | undefined) => (v == null ? '-' : v < 1000 ? `${Math.round(v)} ms` : `${(v / 1000).toFixed(1)} s`)
export const num = (v: number | null | undefined, d = 0) => (v == null ? '-' : v.toFixed(d))
/** Dollars with enough digits to tell one cheap run from another: $0.0045, $1.27. */
export const usd = (v: number | null | undefined) => {
  if (v == null) return '-'
  if (v === 0) return '$0'
  // A cheap hosted run costs fractions of a cent; "$0.0000" would read as free.
  if (v < 0.0001) return '<$0.0001'
  if (v < 0.01) return `$${v.toFixed(4)}`
  return `$${v.toFixed(2)}`
}
export const when = (iso: string) => {
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString(undefined, { dateStyle: 'short', timeStyle: 'short' })
}
export const whenTime = (iso: string) => {
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })
}
export const promptName = (p: { locales: Record<string, { name: string }> }, locale: string) =>
  (p.locales[locale] ?? p.locales.en).name

/** How a file was packaged, not which model it is. Stripped from the right. */
const PACKAGING = /[-_.](gguf|ggml|gptq|awq|exl2|mlx|onnx|safetensors|bf16|fp16|f16|f32|ud|i\d|iq\d[\w]*|q\d[\w]*)$/i

/** Readable default for a model id: org off, packaging noise off. */
export function autoModelLabel(id: string): string {
  let s = id.split('/').pop() ?? id
  // Repeat: "...-i1-GGUF" only loses one token per pass.
  for (let prev = ''; prev !== s; ) {
    prev = s
    s = s.replace(PACKAGING, '')
  }
  return s || id
}

/** Model ids the user renamed by hand. Empty strings are ignored. */
export type Aliases = Record<string, string>

/** Short label for a model: the user's alias when there is one, else the rule above. */
export const modelLabel = (id: string, aliases: Aliases = {}) => aliases[id]?.trim() || autoModelLabel(id)

/** A label with the effort it ran at. Nothing added for default, which is how every run looked before. */
export function withEffort(label: string, effort: string | null | undefined): string {
  if (!effort || effort === 'default') return label
  return `${label} · ${effort === 'off' ? 'thinking off' : `effort ${effort}`}`
}
