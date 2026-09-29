import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, EFFORTS, type Delta, type Effort, type HistoryRow, type MatrixCell, type Meta, type Outcome, type Prompt, type Server } from '../api'
import { promptName, modelLabel } from '../format'
import { extractHtml } from '../html'
import type { Prefs } from '../store'
import { runPool, type Batch, type BatchRow, planRows } from '../batch'
import { BatchView } from './Batch'
import { Library } from './Library'
import { Output } from './Output'
import { Side } from './Side'
import { Starter } from './Starter'

type Props = {
  meta: Meta
  server: Server | null
  prefs: Prefs
  update: (p: Partial<Prefs>) => void
  matrix: MatrixCell[]
  refreshMatrix: () => void
  globalSystem: string
  /** Tells the shell whether something is running, so the header can offer Stop. */
  onRunState: (running: boolean, stop: () => void) => void
}

export function Bench({ meta, server, prefs, update, matrix, refreshMatrix, globalSystem, onRunState }: Props) {
  const prompt = useMemo(() => meta.prompts.find((p) => p.id === prefs.promptId) ?? null, [meta.prompts, prefs.promptId])
  const text = prompt ? (prompt.locales[prefs.locale] ?? prompt.locales.en) : null
  const [system, setSystem] = useState<string>('')
  const [systemTouched, setSystemTouched] = useState(false)
  const [deltas, setDeltas] = useState<Delta[]>([])
  const [outcome, setOutcome] = useState<Outcome | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [running, setRunning] = useState(false)
  const [previous, setPrevious] = useState<HistoryRow[]>([])
  /** Latest runs across every model, for the starter panel. */
  const [recent, setRecent] = useState<HistoryRow[]>([])
  const [viewingId, setViewingId] = useState<number | null>(null)
  const [batch, setBatch] = useState<Batch | null>(null)
  /** Every evaluation in flight. A batch at parallel > 1 has several at once. */
  const aborts = useRef<Set<AbortController>>(new Set())
  const stopAfter = useRef(false)

  // Reset per selection during render (React's "adjusting state on prop change" pattern).
  // The batch survives a prompt switch on purpose: opening a row mid-run must not
  // destroy the running view, the banner above the single prompt leads back to it.
  const selection = `${prefs.promptId ?? ''}|${prefs.locale}`
  const [seen, setSeen] = useState(selection)
  if (seen !== selection) {
    setSeen(selection)
    setSystem(text?.system_prompt ?? '')
    setSystemTouched(false)
    setDeltas([])
    setOutcome(null)
    setError(null)
    setViewingId(null)
  }

  const loadPrevious = useCallback(() => {
    if (!prompt) return
    api
      .history({ prompt_id: prompt.id, limit: 50 })
      .then(setPrevious)
      .catch(() => setPrevious([]))
  }, [prompt])
  useEffect(loadPrevious, [loadPrevious])

  const loadRecent = useCallback(() => {
    api
      .history({ limit: 6 })
      .then(setRecent)
      .catch(() => setRecent([]))
  }, [])
  useEffect(loadRecent, [loadRecent])

  const ready = !!server && !!prefs.model && !running

  async function runPrompt(p: Prompt, collect: (d: Delta) => void): Promise<Outcome> {
    const ctl = new AbortController()
    aborts.current.add(ctl)
    try {
      return await api.evaluate(
        {
          server_id: server!.id,
          model: prefs.model!,
          prompt_id: p.id,
          locale: prefs.locale,
          temperature: prefs.temperature,
          top_p: prefs.topP,
          timeout_s: prefs.timeoutS,
          effort: prefs.effort,
          system_override: p.id === prompt?.id && systemTouched ? system : null,
          parallel: prefs.parallel,
        },
        collect,
        ctl.signal,
      )
    } finally {
      aborts.current.delete(ctl)
    }
  }

  async function run() {
    if (!prompt || !ready) return
    setDeltas([])
    setOutcome(null)
    setError(null)
    setViewingId(null)
    setRunning(true)
    onRunState(true, stopNow)
    try {
      const out = await runPrompt(prompt, (d) => setDeltas((ds) => [...ds, d]))
      setOutcome(out)
      setViewingId(out.result_id)
    } catch (e) {
      if ((e as Error).name !== 'AbortError') setError((e as Error).message)
    } finally {
      setRunning(false)
      onRunState(false, () => {})
      loadPrevious()
      loadRecent()
      refreshMatrix()
    }
  }

  /** Library asks for a batch: show the plan, nothing runs yet. Prompts this model already ran start unticked. */
  function planBatch(ids: string[], label: string) {
    update({ promptId: null })
    setBatch({ label, rows: planRows(ids, matrix, prefs.model, prefs.effort), active: [], started: false })
  }

  async function startBatch() {
    if (!batch || !ready) return
    const ids = batch.rows.filter((r) => r.selected).map((r) => r.id)
    if (!ids.length) return
    const n = Math.max(1, Math.min(prefs.parallel, ids.length))
    stopAfter.current = false
    setBatch((b) => b && { ...b, started: true, active: [], rows: b.rows.map((r) => ({ ...r, status: null, tps: null, detail: '', cost: null })) })
    setRunning(true)
    onRunState(true, stopNow)
    /** One row's result, plus the in-flight list that drives the "running" column. */
    const mark = (id: string, row: Partial<BatchRow> | null) =>
      setBatch((b) =>
        b && {
          ...b,
          active: row ? b.active.filter((x) => x !== id) : [...b.active, id],
          rows: b.rows.map((r) => (r.id === id && row ? { ...r, ...row } : r)),
        },
      )
    await runPool(
      ids,
      n,
      async (id) => {
        const p = meta.prompts.find((x) => x.id === id)!
        mark(id, null)
        // The live pane is one stream, so only a sequential batch feeds it.
        if (n === 1) setDeltas([])
        let row: Partial<BatchRow>
        try {
          const out = await runPrompt(p, n === 1 ? (d) => setDeltas((ds) => [...ds, d]) : () => {})
          row = { status: out.status, tps: out.telemetry?.tps ?? null, detail: out.detail, cost: out.cost_usd }
        } catch (e) {
          if ((e as Error).name === 'AbortError') {
            stopAfter.current = true // Stop aborted every request, not just this one.
            mark(id, { status: 'STOPPED', detail: 'stopped' })
            return
          }
          row = { status: 'ERROR', tps: null, detail: (e as Error).message, cost: null }
        }
        mark(id, row)
      },
      () => stopAfter.current,
    )
    setBatch((b) => b && { ...b, active: [] })
    setRunning(false)
    onRunState(false, () => {})
    loadRecent()
    refreshMatrix()
  }

  function stopNow() {
    for (const ctl of aborts.current) ctl.abort()
    stopAfter.current = true
  }

  async function deletePrevious(row: HistoryRow) {
    if (!window.confirm(`Delete result #${row.id}? The answer, score and trace go with it.`)) return
    try {
      await api.deleteResult(row.id)
      if (viewingId === row.id) {
        setViewingId(null)
        setOutcome(null)
        setDeltas([])
      }
      loadPrevious()
      refreshMatrix()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  async function view(row: HistoryRow) {
    try {
      const r = await api.result(row.id)
      setViewingId(r.id)
      setError(null)
      setDeltas([
        ...(r.reasoning ? [{ kind: 'reasoning' as const, text: r.reasoning }] : []),
        { kind: 'content' as const, text: r.answer },
      ])
      setOutcome({
        result_id: r.id,
        status: r.status,
        detail: r.detail,
        error: r.status === 'ERROR' ? r.detail : null,
        artifact: r.artifact,
        trace: r.trace,
        finish_reason: null,
        answer: r.answer,
        reasoning: r.reasoning,
        tool_calls: [],
        telemetry: {
          ttft_ms: r.ttft_ms,
          total_ms: r.total_ms,
          tokens_in: r.tokens_in,
          tokens_out: r.tokens_out,
          tokens_think: r.tokens_think,
          tps: r.tps,
          usage_from_server: true,
        },
        cost_usd: r.cost_usd,
        cost_source: r.cost_source,
      })
    } catch (e) {
      setError((e as Error).message)
    }
  }

  const artifactSrcDoc = useMemo(() => {
    if (!prompt || prompt.eval_type !== 'render_html' || running) return null
    const answer = deltas
      .filter((d) => d.kind === 'content')
      .map((d) => d.text)
      .join('')
    return answer ? extractHtml(answer) : null
  }, [deltas, prompt, running])

  const runTitle = !server ? 'Add a server first' : !prefs.model ? 'Pick a model first' : ''
  const batchDone = batch ? batch.rows.filter((r) => r.selected && r.status).length : 0
  const batchTotal = batch ? batch.rows.filter((r) => r.selected).length : 0
  const liveId = batch && prefs.parallel === 1 && batch.active.length === 1 ? batch.active[0] : null
  const currentBatchPrompt = liveId ? (meta.prompts.find((p) => p.id === liveId) ?? null) : null

  return (
    <div className="bench">
      <Library
        prompts={meta.prompts}
        categories={meta.categories}
        locale={prefs.locale}
        current={prefs.promptId}
        model={prefs.model}
        matrix={matrix}
        busy={running}
        onPick={(id) => update({ promptId: id })}
        onBatch={planBatch}
      />
      <main className="work">
        {batch && !prompt ? (
          <>
            <BatchView
              batch={batch}
              prompts={meta.prompts}
              categories={meta.categories}
              locale={prefs.locale}
              running={running}
              ready={ready}
              onSelect={(ids, on) => setBatch((b) => b && { ...b, rows: b.rows.map((r) => (ids.includes(r.id) ? { ...r, selected: on } : r)) })}
              onStart={startBatch}
              onStop={stopNow}
              onStopAfter={() => (stopAfter.current = true)}
              onClose={() => setBatch(null)}
              onOpen={(id) => update({ promptId: id })}
              parallel={prefs.parallel}
              onParallel={(n) => update({ parallel: n })}
            />
            {currentBatchPrompt && (
              <section className="live">
                <h2>
                  Now running: {promptName(currentBatchPrompt, prefs.locale)} <code>{currentBatchPrompt.id}</code>
                </h2>
                <Output evalType={currentBatchPrompt.eval_type} deltas={deltas} running={running} artifactSrcDoc={null} />
              </section>
            )}
          </>
        ) : !prompt || !text ? (
          <Starter
            server={server}
            model={prefs.model}
            locale={prefs.locale}
            aliases={prefs.aliases}
            prompts={meta.prompts}
            matrix={matrix}
            recent={recent}
            onOpenSettings={() => update({ view: 'settings' })}
          />
        ) : (
          <>
            {batch && (
              <div className="banner">
                <button className="btn quiet" onClick={() => update({ promptId: null })}>
                  ← Back to batch: {batch.label} ({batchDone}/{batchTotal} done{running ? ', running' : ''})
                </button>
              </div>
            )}
            <header>
              <h1>{promptName(prompt, prefs.locale)}</h1>
              <div className="meta">
                <code>{prompt.id}</code>
                <span>{prompt.eval_type}</span>
                <span>check: {prompt.check.type}</span>
                <span>{prompt.max_tokens ? `max ${prompt.max_tokens} tokens` : 'max tokens: server default'}</span>
                {prompt.tools.length > 0 && <span>tools: {prompt.tools.join(', ')}</span>}
                {prompt.fault && <span>fault: {String(prompt.fault.kind)}</span>}
                {!prompt.locales[prefs.locale] && <span>no {prefs.locale} text, showing en</span>}
              </div>
            </header>
            <div className="prompt-box">
              {globalSystem.trim() && (
                <div className="global-note" title={globalSystem}>
                  Global system prompt is on ({globalSystem.trim().length} chars, edit under Settings). It goes before the text below.
                </div>
              )}
              <details open={systemTouched}>
                <summary>System prompt{systemTouched ? ' (edited)' : system ? '' : ' (empty)'}</summary>
                <textarea
                  rows={3}
                  value={system}
                  onChange={(e) => {
                    setSystem(e.target.value)
                    setSystemTouched(true)
                  }}
                  aria-label="System prompt"
                />
              </details>
              <div className="user" aria-label="User prompt">
                {text.prompt}
              </div>
            </div>
            <div className="runbar">
              {running ? (
                <button className="btn" onClick={stopNow}>
                  Stop
                </button>
              ) : (
                <button className="btn primary" onClick={run} disabled={!ready} title={runTitle || prefs.model || undefined}>
                  Run on {prefs.model ? modelLabel(prefs.model, prefs.aliases) : '...'}
                </button>
              )}
              <label className="param">
                temp
                <input type="number" min={0} max={2} step={0.05} value={prefs.temperature} onChange={(e) => update({ temperature: Number(e.target.value) })} />
              </label>
              <label className="param">
                top-p
                <input type="number" min={0} max={1} step={0.05} value={prefs.topP} onChange={(e) => update({ topP: Number(e.target.value) })} />
              </label>
              <label className="param" title="default sends nothing, so the model thinks as it ships. off turns thinking off. A model that does not take the setting answers with an error.">
                effort
                <select value={prefs.effort} onChange={(e) => update({ effort: e.target.value as Effort })} aria-label="Thinking effort">
                  {EFFORTS.map((x) => (
                    <option key={x} value={x}>
                      {x}
                    </option>
                  ))}
                </select>
              </label>
              <label className="param">
                timeout s
                <input type="number" min={5} max={3600} step={5} value={prefs.timeoutS} onChange={(e) => update({ timeoutS: Number(e.target.value) })} />
              </label>
              <span className="spacer" />
              {viewingId && !running && outcome && <span className="hint">showing result #{viewingId}</span>}
            </div>
            {error && <div className="banner error">{error}</div>}
            {prompt.eval_type === 'error_test' && outcome && !running && (
              <div className={`banner ${outcome.status === 'PASS' ? 'ok' : 'error'}`}>{outcome.detail}</div>
            )}
            <Output evalType={prompt.eval_type} deltas={deltas} running={running} artifactSrcDoc={artifactSrcDoc} />
          </>
        )}
      </main>
      <Side
        prompt={prompt}
        outcome={prompt ? outcome : null}
        running={running && !batch}
        error={prompt ? error : null}
        previous={previous}
        viewingId={viewingId}
        aliases={prefs.aliases}
        onView={view}
        onDelete={deletePrevious}
        onScored={loadPrevious}
      />
    </div>
  )
}
