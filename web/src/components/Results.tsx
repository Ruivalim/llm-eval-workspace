import { useEffect, useMemo, useState } from 'react'
import { api, artifactUrl, traceUrl, type HistoryRow, type LeaderboardRow, type MatrixCell, type Meta, type ResultRow } from '../api'
import { ms, modelLabel, num, promptName, usd, when, withEffort, type Aliases } from '../format'
import { renderMarkdown } from '../markdown'
import { Tele } from './Side'

const STATUSES = ['PASS', 'FAIL', 'ERROR']
const PAGE_SIZE = 50

export function Results({ meta, locale, matrix, aliases, onChange }: { meta: Meta; locale: string; matrix: MatrixCell[]; aliases: Aliases; onChange: () => void }) {
  const [lb, setLb] = useState<LeaderboardRow[]>([])
  const [rows, setRows] = useState<HistoryRow[]>([])
  const [filter, setFilter] = useState({ model: '', prompt_id: '', status: '', locale: '' })
  const [open, setOpen] = useState<ResultRow | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [deleting, setDeleting] = useState(false)
  const [deleteModel, setDeleteModel] = useState('')
  const [tick, setTick] = useState(0)
  const [compare, setCompare] = useState<MatrixCell[]>(matrix)
  const [hiddenModels, setHiddenModels] = useState<string[]>([])
  const [page, setPage] = useState(0)

  useEffect(() => {
    const context = { model: filter.model, prompt_id: filter.prompt_id, locale: filter.locale }
    api.leaderboard(context).then(setLb).catch((e) => setErr((e as Error).message))
    api.matrix(context).then(setCompare).catch((e) => setErr((e as Error).message))
  }, [filter, matrix, tick])
  function updateFilter(patch: Partial<typeof filter>) {
    setFilter({ ...filter, ...patch })
    setPage(0)
  }
  useEffect(() => {
    api
      .history({ ...filter, limit: 300 })
      .then(setRows)
      .catch((e) => setErr((e as Error).message))
  }, [filter, matrix, tick])

  const modelOptions = useMemo(() => Array.from(new Set(matrix.map((c) => c.model))).sort(), [matrix])
  /** A matrix column is a model plus server plus locale. The key stays raw, only the label shortens. */
  const cols = useMemo(() => {
    const m = new Map<string, string>()
    for (const c of compare)
      m.set(`${c.model} · ${c.effort} · server ${c.server_id} · ${c.locale}`, `${withEffort(modelLabel(c.model, aliases), c.effort)} · server ${c.server_id} · ${c.locale}`)
    return [...m.entries()].sort((a, b) => a[0].localeCompare(b[0]))
  }, [compare, aliases])
  const visibleCols = useMemo(() => cols.filter(([k]) => !hiddenModels.includes(k)), [cols, hiddenModels])
  const toggleModel = (m: string) =>
    setHiddenModels((h) => (h.includes(m) ? h.filter((x) => x !== m) : [...h, m]))
  const pageCount = Math.max(1, Math.ceil(rows.length / PAGE_SIZE))
  const safePage = Math.min(page, pageCount - 1)
  const paged = useMemo(() => rows.slice(safePage * PAGE_SIZE, safePage * PAGE_SIZE + PAGE_SIZE), [rows, safePage])
  const cell = useMemo(() => {
    const m = new Map<string, MatrixCell>()
    for (const c of compare) m.set(`${c.model} · ${c.effort} · server ${c.server_id} · ${c.locale} ${c.prompt_id}`, c)
    return m
  }, [compare])
  const name = (id: string) => {
    const p = meta.prompts.find((x) => x.id === id)
    return p ? promptName(p, locale) : id
  }
  async function show(id: number) {
    try {
      setOpen(await api.result(id))
    } catch (e) {
      setErr((e as Error).message)
    }
  }

  /** Every deletion writes to the same tables the matrix and leaderboard read. */
  function reload() {
    setTick((t) => t + 1)
    onChange()
  }

  async function del(id: number) {
    if (!window.confirm(`Delete result #${id}? The answer, score and trace go with it.`)) return
    setDeleting(true)
    try {
      await api.deleteResult(id)
      setOpen((o) => (o?.id === id ? null : o))
      setErr(null)
      reload()
    } catch (e) {
      setErr((e as Error).message)
    } finally {
      setDeleting(false)
    }
  }

  /** Delete every result saved for one model, nothing else. */
  async function deleteModelResults() {
    if (!deleteModel) return
    const runs = lb.find((r) => r.model === deleteModel)?.runs
    const detail = runs != null ? `${runs} result(s) from "${deleteModel}"` : `all results from "${deleteModel}"`
    if (!window.confirm(`Delete ${detail}? Prompts and servers stay. This cannot be undone.`)) return
    setDeleting(true)
    try {
      await api.clearResults({ model: deleteModel })
      setOpen(null)
      setErr(null)
      setDeleteModel('')
      reload()
    } catch (e) {
      setErr((e as Error).message)
    } finally {
      setDeleting(false)
    }
  }

  return (
    <div className="page">
      <h1>Results</h1>
      {err && <div className="banner error">{err}</div>}

      {lb.length > 0 && (
        <section>
          <h2>
            Per model<small>same model, prompt and locale filters as matrix</small>
          </h2>
          <div className="chart" role="img" aria-label="Pass rate per model">
            {lb.map((r) => (
              <div className="chart-row" key={`${r.model} ${r.effort}`}>
                <span className="chart-label" title={r.model}>{withEffort(modelLabel(r.model, aliases), r.effort)}</span>
                <div className="chart-track">
                  <i className="chart-fill" style={{ width: `${Math.min(100, Math.max(0, r.pass_rate))}%` }} />
                </div>
                <span className="chart-val mono">{num(r.pass_rate, 1)}% ({r.passed}/{r.prompts})</span>
              </div>
            ))}
          </div>
          <table className="grid">
            <thead>
              <tr>
                <th>model</th>
                <th className="num">runs</th>
                <th className="num">prompts</th>
                <th className="num">pass rate</th>
                <th className="num">avg TTFT</th>
                <th className="num">avg t/s</th>
                <th className="num">avg human</th>
                <th className="num">spent</th>
              </tr>
            </thead>
            <tbody>
              {lb.map((r) => (
                <tr key={`${r.model} ${r.effort}`}>
                  <td title={r.model}>{withEffort(modelLabel(r.model, aliases), r.effort)}</td>
                  <td className="num">{r.runs}</td>
                  <td className="num">{r.prompts}</td>
                  <td className="num">{num(r.pass_rate, 1)}%</td>
                  <td className="num">{ms(r.avg_ttft_ms)}</td>
                  <td className="num">{num(r.avg_tps, 1)}</td>
                  <td className="num">{r.avg_human ?? '-'}</td>
                  <td
                    className="num"
                    title={r.priced < r.runs ? `${r.runs - r.priced} of ${r.runs} results have no known price and are not counted` : undefined}
                  >
                    {usd(r.total_cost)}
                    {r.total_cost != null && r.priced < r.runs && <span className="dim">*</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      <section className="matrix-section">
        <h2>
          Latest result per prompt and model<small>click a cell to open it</small>
        </h2>
        {cols.length > 0 && (
          <fieldset className="model-filter">
            <legend>models shown</legend>
            {cols.map(([k, label]) => (
              <label key={k} title={k}>
                <input type="checkbox" checked={!hiddenModels.includes(k)} onChange={() => toggleModel(k)} />
                {label}
              </label>
            ))}
          </fieldset>
        )}
        {cols.length === 0 || visibleCols.length === 0 ? (
          <div className="empty">
            <b>{cols.length === 0 ? 'Nothing run yet.' : 'No models selected.'}</b>{cols.length === 0 ? 'Run a prompt on the Bench and it lands here.' : 'Tick a model above to show its column.'}
          </div>
        ) : (
          <div className="matrix">
            <table>
              <thead>
                <tr>
                  <th>prompt</th>
                  {visibleCols.map(([k, label]) => (
                    <th key={k} title={k}>{label}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {meta.categories.flatMap((cat) =>
                  meta.prompts
                    .filter((p) => p.category === cat.slug)
                    .map((p) => (
                      <tr key={p.id}>
                        <td title={name(p.id)}>{p.id}</td>
                        {visibleCols.map(([k]) => {
                          const c = cell.get(`${k} ${p.id}`)
                          return (
                            <td key={k} className="cell">
                              {c ? (
                                <button onClick={() => show(c.id)} title={c.detail}>
                                  <span className={`status ${c.status}`}>{c.status}</span>
                                  <span className="tps">
                                    {num(c.tps, 0)} t/s{c.human_score != null ? ` / ${c.human_score}/10` : ''}
                                  </span>
                                </button>
                              ) : (
                                <span className="none">.</span>
                              )}
                            </td>
                          )
                        })}
                      </tr>
                    )),
                )}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section>
        <h2>
          All runs<small>{rows.length} total · page {safePage + 1} of {pageCount}</small>
        </h2>
        <div className="filters">
          <label className="stack">
            model
            <select value={filter.model} onChange={(e) => updateFilter({ model: e.target.value })}>
              <option value="">any</option>
              {modelOptions.map((m) => (
                <option key={m} value={m} title={m}>
                  {modelLabel(m, aliases)}
                </option>
              ))}
            </select>
          </label>
          <label className="stack">
            prompt
            <select value={filter.prompt_id} onChange={(e) => updateFilter({ prompt_id: e.target.value })}>
              <option value="">any</option>
              {meta.prompts.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.id}
                </option>
              ))}
            </select>
          </label>
          <label className="stack">
            status
            <select value={filter.status} onChange={(e) => updateFilter({ status: e.target.value })}>
              <option value="">any</option>
              {STATUSES.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </label>
          <label className="stack">
            locale
            <select value={filter.locale} onChange={(e) => updateFilter({ locale: e.target.value })}>
              <option value="">any</option>
              {meta.locales.map((l) => (
                <option key={l} value={l}>
                  {l}
                </option>
              ))}
            </select>
          </label>
        </div>
        <div className="filters">
          <label className="stack">
            Delete all results from model
            <select value={deleteModel} onChange={(e) => setDeleteModel(e.target.value)} aria-label="Model to delete">
              <option value="">select model</option>
              {modelOptions.map((m) => (
                <option key={m} value={m} title={m}>
                  {modelLabel(m, aliases)}
                </option>
              ))}
            </select>
          </label>
          <button
            type="button"
            className="btn danger"
            onClick={deleteModelResults}
            disabled={deleting || !deleteModel}
            title="Delete every saved result for the selected model"
          >
            {deleting ? 'deleting…' : 'delete model results'}
          </button>
        </div>
        {rows.length === 0 ? (
          <div className="empty">
            <b>No results match.</b>Loosen the filters or run something.
          </div>
        ) : (
          <table className="grid">
            <thead>
              <tr>
                <th>when</th>
                <th>model</th>
                <th>prompt</th>
                <th>status</th>
                <th className="num">TTFT</th>
                <th className="num">t/s</th>
                <th className="num">out</th>
                <th className="num">cost</th>
                <th className="num">score</th>
                <th>detail</th>
              </tr>
            </thead>
            <tbody>
              {paged.map((r) => (
                <tr key={r.id} className="click" onClick={() => show(r.id)}>
                  <td className="dim">{when(r.created_at)}</td>
                  <td title={r.model}>{withEffort(modelLabel(r.model, aliases), r.effort)}</td>
                  <td>
                    {r.prompt_id}
                    <span className="dim"> {r.locale}</span>
                  </td>
                  <td>
                    <span className={`status ${r.status}`}>{r.status}</span>
                  </td>
                  <td className="num">{ms(r.ttft_ms)}</td>
                  <td className="num" title={r.parallel > 1 ? `${r.parallel} requests in flight: t/s is not comparable with a sequential run` : undefined}>
                    {num(r.tps, 0)}
                    {r.parallel > 1 && <span className="dim"> ×{r.parallel}</span>}
                  </td>
                  <td className="num">{r.tokens_out}</td>
                  <td className="num">{usd(r.cost_usd)}</td>
                  <td className="num">{r.human_score ?? '-'}</td>
                  <td className="dim" title={r.detail}>
                    {r.detail.slice(0, 80)}
                  </td>
                  <td>
                    <button
                      className="btn quiet danger"
                      disabled={deleting}
                      title={`Delete result #${r.id}`}
                      onClick={(e) => {
                        e.stopPropagation()
                        del(r.id)
                      }}
                    >
                      delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {pageCount > 1 && (
          <div className="pager">
            <button type="button" className="btn" disabled={safePage === 0} onClick={() => setPage(safePage - 1)}>
              ← prev
            </button>
            <span className="dim">page {safePage + 1} of {pageCount}</span>
            <button type="button" className="btn" disabled={safePage >= pageCount - 1} onClick={() => setPage(safePage + 1)}>
              next →
            </button>
          </div>
        )}
      </section>

      {open && <Drawer row={open} name={name(open.prompt_id)} aliases={aliases} onClose={() => setOpen(null)} onDelete={() => del(open.id)} />}
    </div>
  )
}

function Drawer({ row, name, aliases, onClose, onDelete }: { row: ResultRow; name: string; aliases: Aliases; onClose: () => void; onDelete: () => void }) {
  useEffect(() => {
    const k = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', k)
    return () => window.removeEventListener('keydown', k)
  }, [onClose])
  const raw = row.eval_type === 'render_html' || row.eval_type === 'tool_call'
  return (
    <div className="drawer-bg" onClick={onClose}>
      <div className="drawer" role="dialog" aria-modal="true" aria-label={`Result ${row.id}`} onClick={(e) => e.stopPropagation()}>
        <header>
          <span className={`status ${row.status}`}>{row.status}</span>
          <h2>{name}</h2>
          <button className="btn danger" onClick={onDelete}>
            Delete
          </button>
          <button className="btn quiet" onClick={onClose}>
            Close
          </button>
        </header>
        <div className="kv">
          <div>
            <span>model</span>
            <b title={row.model}>{modelLabel(row.model, aliases)}</b>
          </div>
          <div>
            <span>prompt</span>
            <b>{row.prompt_id}</b>
          </div>
          <div>
            <span>locale</span>
            <b>{row.locale}</b>
          </div>
          <div>
            <span>when</span>
            <b>{when(row.created_at)}</b>
          </div>
          <div>
            <span>temperature</span>
            <b>{row.temperature ?? '-'}</b>
          </div>
          <div>
            <span>effort</span>
            <b>{row.effort}</b>
          </div>
          <div title={row.parallel > 1 ? 'Other prompts were running at the same time, so t/s is not comparable with a sequential run' : 'Nothing else was running' }>
            <span>parallel</span>
            <b>{row.parallel}</b>
          </div>
          <div>
            <span>result id</span>
            <b>{row.id}</b>
          </div>
        </div>
        <div className="banner">{row.detail}</div>
        <Tele
          t={{
            ttft_ms: row.ttft_ms,
            total_ms: row.total_ms,
            tokens_in: row.tokens_in,
            tokens_out: row.tokens_out,
            tokens_think: row.tokens_think,
            tps: row.tps,
            usage_from_server: true,
          }}
          cost={row}
        />
        {row.human_score != null && (
          <div>
            Human score <b className="mono">{row.human_score}/10</b>
            {row.human_comment ? `: ${row.human_comment}` : ''}
          </div>
        )}
        <div className="links">
          {row.artifact && (
            <a href={artifactUrl(row.artifact)} target="_blank" rel="noreferrer">
              Open rendered artifact
            </a>
          )}
          {row.trace && (
            <a href={traceUrl(row.id, true)} target="_blank" rel="noreferrer" title="Every chunk the server sent, as JSON Lines">
              Download trace
            </a>
          )}
        </div>
        {row.reasoning && (
          <details className="think">
            <summary>Reasoning</summary>
            <pre>{row.reasoning}</pre>
          </details>
        )}
        {raw ? <pre className="code">{row.answer}</pre> : <div className="answer" dangerouslySetInnerHTML={{ __html: renderMarkdown(row.answer) }} />}
      </div>
    </div>
  )
}
