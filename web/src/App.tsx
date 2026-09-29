import { useCallback, useEffect, useState } from 'react'
import { api, type AppSettings, type MatrixCell, type Meta, type Model, type Server } from './api'
import { Bench } from './components/Bench'
import { Results } from './components/Results'
import { Settings } from './components/Settings'
import { TopBar } from './components/TopBar'
import { usePrefs } from './store'

export default function App() {
  const [prefs, update] = usePrefs()
  const [meta, setMeta] = useState<Meta | null>(null)
  const [servers, setServers] = useState<Server[]>([])
  // Models are keyed by the server they were fetched for; a mismatch means "loading".
  const [modelsFor, setModelsFor] = useState<{ serverId: number | null; list: Model[]; error: string | null }>({ serverId: null, list: [], error: null })
  const [modelsTick, setModelsTick] = useState(0)
  const [matrix, setMatrix] = useState<MatrixCell[]>([])
  const [fatal, setFatal] = useState<string | null>(null)
  const [settings, setSettings] = useState<AppSettings>({ global_system_prompt: '' })
  const [run, setRun] = useState<{ running: boolean; stop: () => void }>({ running: false, stop: () => {} })
  const onRunState = useCallback((running: boolean, stop: () => void) => setRun({ running, stop }), [])

  const refreshServers = useCallback(
    () =>
      api
        .servers()
        .then(setServers)
        .catch((e) => setFatal((e as Error).message)),
    [],
  )
  const refreshMatrix = useCallback(
    () =>
      api
        .matrix()
        .then(setMatrix)
        .catch(() => {}),
    [],
  )

  useEffect(() => {
    api
      .meta()
      .then(setMeta)
      .catch((e) => setFatal((e as Error).message))
    refreshServers()
    refreshMatrix()
    api
      .settings()
      .then(setSettings)
      .catch(() => {})
  }, [refreshServers, refreshMatrix])

  // A stale or missing selection falls back to the first server without a re-render.
  const server = servers.find((s) => s.id === prefs.serverId) ?? servers[0] ?? null

  const serverId = server?.id ?? null
  useEffect(() => {
    if (serverId === null) return
    let alive = true
    api
      .models(serverId)
      .then((list) => alive && setModelsFor({ serverId, list, error: null }))
      .catch((e) => alive && setModelsFor({ serverId, list: [], error: (e as Error).message }))
    return () => {
      alive = false
    }
  }, [serverId, modelsTick])
  const refreshModels = useCallback(() => {
    setModelsFor({ serverId: null, list: [], error: null })
    setModelsTick((t) => t + 1)
  }, [])
  const models = serverId !== null && modelsFor.serverId === serverId ? modelsFor.list : []
  const modelsError = modelsFor.serverId === serverId ? modelsFor.error : null
  const loadingModels = serverId !== null && modelsFor.serverId !== serverId

  if (fatal)
    return (
      <div className="empty">
        <b>The API is not reachable.</b>
        {fatal}. Start it with <code>make api</code>.
      </div>
    )
  if (!meta) return <div className="empty">Loading</div>

  return (
    <div className="shell">
      <TopBar
        servers={servers}
        prefs={prefs}
        update={update}
        locales={meta.locales}
        models={models}
        modelsError={modelsError}
        loadingModels={loadingModels}
        refreshModels={refreshModels}
        running={run.running}
        onStop={run.stop}
      />
      {prefs.view === 'bench' && (
        <Bench meta={meta} server={server} prefs={prefs} update={update} matrix={matrix} refreshMatrix={refreshMatrix} globalSystem={settings.global_system_prompt} onRunState={onRunState} />
      )}
      {prefs.view === 'results' && <Results meta={meta} locale={prefs.locale} matrix={matrix} aliases={prefs.aliases} onChange={refreshMatrix} />}
      {prefs.view === 'settings' && (
        <Settings
          servers={servers}
          onServersChange={refreshServers}
          settings={settings}
          onSettingsChange={setSettings}
          models={models}
          aliases={prefs.aliases}
          onAliasesChange={(aliases) => update({ aliases })}
        />
      )}
    </div>
  )
}
