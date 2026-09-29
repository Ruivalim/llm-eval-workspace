import { useState } from 'react'
import { api, artifactUrl, traceUrl, type Cost, type HistoryRow, type Outcome, type Prompt, type Telemetry } from '../api'
import { modelLabel, ms, num, usd, when, whenTime, type Aliases } from '../format'

type Props = {
  prompt: Prompt | null
  outcome: Outcome | null
  running: boolean
  error: string | null
  previous: HistoryRow[]
  viewingId: number | null
  aliases: Aliases
  onView: (row: HistoryRow) => void
  onDelete: (row: HistoryRow) => void
  onScored: () => void
}

export function Side({ prompt, outcome, running, error, previous, viewingId, aliases, onView, onDelete, onScored }: Props) {
  const status = running ? 'RUNNING' : outcome?.status ?? (error ? 'ERROR' : 'NONE')
  return (
    <aside className="side" aria-label="Verdict and telemetry">
      <section className="verdict" aria-live="polite">
        <h2>Verdict</h2>
        <div className={`big ${status}`}>{status === 'NONE' ? '—' : status}</div>
        <div className="detail">
          {running ? 'Streaming…' : outcome?.detail ?? error ?? (prompt ? `Check: ${prompt.check.type}` : '')}
        </div>
        {outcome?.error && !running && prompt?.eval_type !== 'error_test' && <div className="banner error">{outcome.error}</div>}
        {outcome?.finish_reason === 'length' && !running && <div className="banner">Cut off by max_tokens: the model wanted to keep going.</div>}
        {outcome?.finish_reason === 'loop' && !running && <div className="banner">Stopped early: the model entered a repetition loop. Partial answer kept.</div>}
        <div className="links">
          {outcome?.artifact && (
            <a href={artifactUrl(outcome.artifact)} target="_blank" rel="noreferrer">
              Open rendered artifact
            </a>
          )}
          {outcome?.result_id && outcome.trace && !running && (
            <a href={traceUrl(outcome.result_id, true)} target="_blank" rel="noreferrer" title="Every chunk the server sent, as JSON Lines">
              Download trace
            </a>
          )}
        </div>
      </section>
      <section>
        <h2>Telemetry</h2>
        {outcome?.telemetry ? <Tele t={outcome.telemetry} cost={outcome} /> : <div className="detail" style={{ color: 'var(--fg-3)', fontSize: 13 }}>Numbers appear after the run.</div>}
      </section>
      {outcome?.result_id && !running && <Score resultId={outcome.result_id} key={outcome.result_id} onScored={onScored} />}
      {prompt && (
        <section>
          <h2>Previous runs · {prompt.id}</h2>
          {previous.length === 0 ? (
            <div style={{ color: 'var(--fg-3)', fontSize: 13 }}>No runs of this prompt yet.</div>
          ) : (
            /* A row per run: verdict, model, when, and t/s with the score. Enough to
               compare two old runs without opening them one by one. */
            <div className="prev">
              {previous.slice(0, 12).map((r) => (
                <div className="prev-row" key={r.id}>
                  <button
                    className="pick-row"
                    aria-current={viewingId === r.id ? 'true' : undefined}
                    onClick={() => onView(r)}
                    title={`${when(r.created_at)} · ${r.model} · ${r.detail}`}
                  >
                    <span className={`status ${r.status}`}>{r.status}</span>
                    <span className="m">{modelLabel(r.model, aliases)}</span>
                    <span className="when">{whenTime(r.created_at)}</span>
                    <span className="sub">
                      {num(r.tps, 0)} t/s{r.human_score != null ? ` · ${r.human_score}/10` : ''}
                    </span>
                  </button>
                  <button className="btn quiet danger" onClick={() => onDelete(r)} aria-label={`Delete result ${r.id}`} title="Delete this run">
                    ×
                  </button>
                </div>
              ))}
            </div>
          )}
        </section>
      )}
    </aside>
  )
}

const COST_SOURCE: Record<string, string> = {
  server: 'billed by the endpoint',
  price: 'tokens × your price in Settings',
  builtin: 'tokens × Anthropic list price',
}

export function Tele({ t, cost }: { t: Telemetry; cost?: Cost }) {
  return (
    <dl className="tele">
      <div className="hero" style={{ display: 'contents' }}>
        <dt>Tokens / s</dt>
        <dd>{num(t.tps, 1)}</dd>
      </div>
      <dt>Time to first token</dt>
      <dd>{ms(t.ttft_ms)}</dd>
      <dt>Total</dt>
      <dd>{ms(t.total_ms)}</dd>
      <dt>Tokens out</dt>
      <dd>{t.tokens_out}</dd>
      <dt>Thinking tokens</dt>
      <dd>{t.tokens_think}</dd>
      <dt>Tokens in</dt>
      <dd>{t.tokens_in}</dd>
      {cost && (
        <>
          <dt>Cost</dt>
          <dd title={cost.cost_usd == null ? 'No price for this model. Set one in Settings.' : COST_SOURCE[cost.cost_source]}>
            {cost.cost_usd == null ? 'unknown' : usd(cost.cost_usd)}
          </dd>
        </>
      )}
      {!t.usage_from_server && <div className="note">Counts estimated with tiktoken; the server sent no usage.</div>}
    </dl>
  )
}

function Score({ resultId, onScored }: { resultId: number; onScored: () => void }) {
  const [score, setScore] = useState<number | null>(null)
  const [comment, setComment] = useState('')
  const [saved, setSaved] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  async function save() {
    if (score === null) return
    try {
      await api.score(resultId, score, comment)
      setSaved(true)
      setErr(null)
      onScored()
    } catch (e) {
      setErr((e as Error).message)
    }
  }
  return (
    <section className="score">
      <h2>Your score</h2>
      <div className="scale" role="radiogroup" aria-label="Human score 0 to 10">
        {Array.from({ length: 11 }, (_, i) => (
          <button key={i} role="radio" aria-checked={score === i} aria-pressed={score === i} onClick={() => { setScore(i); setSaved(false) }}>
            {i}
          </button>
        ))}
      </div>
      <div className="row">
        <input type="text" placeholder="Why (optional)" value={comment} onChange={(e) => { setComment(e.target.value); setSaved(false) }} aria-label="Comment" />
        <button className="btn" onClick={save} disabled={score === null}>
          {saved ? 'Saved' : 'Save'}
        </button>
      </div>
      {err && <div className="banner error">{err}</div>}
    </section>
  )
}
