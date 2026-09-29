import { useEffect, useState } from 'react'
import { api, type Model, type Server } from '../api'
import { modelLabel } from '../format'
import type { Prefs } from '../store'

type Props = {
  servers: Server[]
  prefs: Prefs
  update: (p: Partial<Prefs>) => void
  locales: string[]
  models: Model[]
  modelsError: string | null
  loadingModels: boolean
  refreshModels: () => void
  running: boolean
  onStop: () => void
}

export function TopBar({ servers, prefs, update, locales, models, modelsError, loadingModels, refreshModels, running, onStop }: Props) {
  const server = servers.find((s) => s.id === prefs.serverId) ?? null
  const loaded = models.find((m) => m.loaded)?.id ?? null
  const [loading, setLoading] = useState<string | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)

  useEffect(() => {
    if (models.length && (!prefs.model || !models.some((m) => m.id === prefs.model))) {
      update({ model: loaded ?? models[0].id })
    }
  }, [models, loaded, prefs.model, update])

  async function load() {
    if (!server || !prefs.model) return
    setLoading(prefs.model)
    setLoadError(null)
    try {
      await api.loadModel(server.id, prefs.model)
      refreshModels()
    } catch (e) {
      setLoadError((e as Error).message)
    } finally {
      setLoading(null)
    }
  }

  const needsLoad = server?.flavor === 'unsloth' && prefs.model && prefs.model !== loaded

  return (
    <div className="top">
      <div className="brand">
        LLM Eval Workspace<small>local bench</small>
      </div>
      <div className="field">
        <span>Server</span>
        <select
          value={prefs.serverId ?? ''}
          onChange={(e) => update({ serverId: e.target.value ? Number(e.target.value) : null, model: null })}
          aria-label="Server"
          disabled={running}
        >
          {servers.length === 0 && <option value="">none configured</option>}
          {servers.map((s) => (
            <option key={s.id} value={s.id}>
              {s.name}
            </option>
          ))}
        </select>
      </div>
      <div className="field">
        <span>Model</span>
        <select
          value={prefs.model ?? ''}
          onChange={(e) => update({ model: e.target.value || null })}
          disabled={!models.length || running}
          aria-label="Model"
          title={modelsError ?? undefined}
        >
          {loadingModels && <option value="">listing…</option>}
          {!loadingModels && !models.length && <option value="">{modelsError ? 'unavailable' : 'no models'}</option>}
          {models.map((m) => (
            <option key={m.id} value={m.id} title={m.id}>
              {modelLabel(m.id, prefs.aliases)}
              {m.loaded ? '  (active)' : ''}
            </option>
          ))}
        </select>
        <button className="btn quiet" onClick={refreshModels} disabled={loadingModels || running} aria-label="Reload models" title="Fetch the model list from the server again">
          Reload models
        </button>
        {needsLoad && (
          <button className="btn" onClick={load} disabled={loading !== null} title={`Active on server: ${loaded ?? 'none'}`}>
            {loading ? 'Loading…' : 'Load on server'}
          </button>
        )}
        {(modelsError || loadError) && (
          <>
            <span className="top-error">{modelsError ?? loadError}</span>
            <button className="btn quiet" onClick={refreshModels}>
              retry
            </button>
          </>
        )}
      </div>
      <div className="field">
        <span>Locale</span>
        <div className="seg" role="group" aria-label="Prompt locale">
          {locales.map((l) => (
            <button key={l} aria-pressed={prefs.locale === l} onClick={() => update({ locale: l })} disabled={running}>
              {l}
            </button>
          ))}
        </div>
      </div>
      {running && (
        <button className="btn stop" onClick={onStop} title="Abort the current request; a batch stops after it">
          <span className="dot RUNNING" /> Stop
        </button>
      )}
      <nav className="nav" aria-label="Views">
        {(['bench', 'results', 'settings'] as const).map((v) => (
          <button key={v} aria-current={prefs.view === v ? 'page' : undefined} onClick={() => update({ view: v })} disabled={running && v !== 'bench'}>
            {v === 'bench' ? 'Bench' : v === 'results' ? 'Results' : 'Settings'}
          </button>
        ))}
      </nav>
    </div>
  )
}
