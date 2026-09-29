/** Selections that should survive a reload. Nothing else is persisted. */
import { useCallback, useState } from 'react'
import { EFFORTS, type Effort } from './api'

const KEY = 'lew.prefs.v1'
export type Prefs = {
  serverId: number | null
  model: string | null
  locale: string
  promptId: string | null
  temperature: number
  topP: number
  timeoutS: number
  /** Thinking effort sent with each run; default sends nothing. */
  effort: Effort
  /** How many batch prompts run at once. 1 = the old sequential behaviour. */
  parallel: number
  /** Hand-renamed model labels, keyed by model id. Missing = use the auto rule. */
  aliases: Record<string, string>
  view: 'bench' | 'results' | 'settings'
}
const DEFAULTS: Prefs = {
  serverId: null, model: null, locale: 'en', promptId: null, temperature: 0, topP: 1, timeoutS: 600, effort: 'default', parallel: 1, aliases: {}, view: 'bench',
}

export function loadPrefs(): Prefs {
  try {
    const raw = localStorage.getItem(KEY)
    const parsed: Record<string, unknown> = raw ? (JSON.parse(raw) as Record<string, unknown>) : {}
    if (parsed.view === 'servers') parsed.view = 'settings' // renamed view, keep old prefs working
    const merged = { ...DEFAULTS, ...(parsed as Partial<Prefs>) }
    // Never let a prefs file from an older build leave aliases undefined.
    const a = merged.aliases
    merged.aliases = a && typeof a === 'object' && !Array.isArray(a) ? { ...(a as Record<string, string>) } : {}
    // A level this build does not know (typo, removed option) falls back to default.
    if (!(EFFORTS as readonly string[]).includes(merged.effort)) merged.effort = 'default'
    return merged
  } catch {
    return { ...DEFAULTS, aliases: {} }
  }
}

export function usePrefs(): [Prefs, (patch: Partial<Prefs>) => void] {
  const [prefs, setPrefs] = useState<Prefs>(loadPrefs)
  const update = useCallback((patch: Partial<Prefs>) => {
    setPrefs((p) => {
      const next = { ...p, ...patch }
      try {
        localStorage.setItem(KEY, JSON.stringify(next))
      } catch {
        /* private mode */
      }
      return next
    })
  }, [])
  return [prefs, update]
}

const FOLD_KEY = 'lew.library.closed.v1'

/** Slugs the user collapsed. Survives reload; nothing else here is persisted. */
export function loadClosed(): Set<string> {
  try {
    const raw = localStorage.getItem(FOLD_KEY)
    const parsed: unknown = raw ? JSON.parse(raw) : []
    return new Set(Array.isArray(parsed) ? parsed.filter((x): x is string => typeof x === 'string') : [])
  } catch {
    return new Set()
  }
}

export function saveClosed(closed: Set<string>): void {
  try {
    localStorage.setItem(FOLD_KEY, JSON.stringify([...closed]))
  } catch {
    /* private mode */
  }
}
