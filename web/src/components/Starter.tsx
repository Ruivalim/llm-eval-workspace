import type { HistoryRow, MatrixCell, Prompt, Server } from '../api'
import { modelLabel, num, promptName, when, type Aliases } from '../format'

type Props = {
  server: Server | null
  model: string | null
  locale: string
  aliases: Aliases
  prompts: Prompt[]
  matrix: MatrixCell[]
  /** Most recent runs across every model, newest first. */
  recent: HistoryRow[]
  onOpenSettings: () => void
}

/** What the bench shows before a prompt is picked. Used to be a bare "Pick a prompt
 *  on the left", which left the whole centre empty and said nothing about how to start. */
export function Starter({ server, model, locale, aliases, prompts, matrix, recent, onOpenSettings }: Props) {
  const byId = new Map(prompts.map((p) => [p.id, p]))
  const mine = model ? matrix.filter((c) => c.model === model) : []
  const passed = mine.filter((c) => c.status === 'PASS').length
  const name = (id: string) => {
    const p = byId.get(id)
    return p ? promptName(p, locale) : id
  }
  return (
    <section className="starter">
      <h1>Pick a prompt on the left</h1>
      <p className="lede">
        The left column is a fixed battery: same prompts, one automatic check each, run against any OpenAI-compatible endpoint or Claude. Watch it stream,
        and every run stays in Results for comparison.
      </p>
      {!server ? (
        <div className="banner">
          <b>No server configured yet.</b> Add one under Settings. Any OpenAI-compatible base URL works, for example{' '}
          <code>http://localhost:11434/v1</code> for Ollama, <code>http://127.0.0.1:8080/v1</code> for llama.cpp, or a hosted one like{' '}
          <code>https://api.deepseek.com/v1</code>. For Claude, pick the <code>anthropic</code> flavor.
        </div>
      ) : !model ? (
        <div className="banner">
          <b>No model selected.</b> Pick one in the header, next to the server name.
        </div>
      ) : (
        <div className="starter-stats">
          <div>
            <span>Model</span>
            <b title={model}>{modelLabel(model, aliases)}</b>
          </div>
          <div>
            <span>Passed</span>
            <b>
              {passed} of {prompts.length} prompts
            </b>
          </div>
          <div>
            <span>Saved runs</span>
            <b>{mine.length} on this model</b>
          </div>
        </div>
      )}
      {recent.length > 0 && (
        <>
          <h2>Last runs</h2>
          <table className="grid">
            <thead>
              <tr>
                <th>when</th>
                <th>model</th>
                <th>prompt</th>
                <th>status</th>
                <th className="num">t/s</th>
              </tr>
            </thead>
            <tbody>
              {recent.map((r) => (
                <tr key={r.id}>
                  <td className="dim">{when(r.created_at)}</td>
                  <td title={r.model}>{modelLabel(r.model, aliases)}</td>
                  <td title={r.prompt_id}>{name(r.prompt_id)}</td>
                  <td>
                    <span className={`status ${r.status}`}>{r.status}</span>
                  </td>
                  <td className="num">{num(r.tps, 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
      {!server && (
        <div>
          <button className="btn primary" onClick={onOpenSettings}>
            Open Settings
          </button>
        </div>
      )}
    </section>
  )
}
