import type { Category, Prompt } from '../api'
import { batchCost, clampParallel, groupRows, MAX_PARALLEL, settled, type Batch } from '../batch'
import { promptName, usd } from '../format'

export function BatchView({
  batch,
  prompts,
  categories,
  locale,
  running,
  ready,
  onSelect,
  onStart,
  onStop,
  onStopAfter,
  onClose,
  onOpen,
  parallel,
  onParallel,
}: {
  batch: Batch
  prompts: Prompt[]
  categories: Category[]
  locale: string
  running: boolean
  ready: boolean
  onSelect: (ids: string[], on: boolean) => void
  onStart: () => void
  onStop: () => void
  onStopAfter: () => void
  onClose: () => void
  onOpen: (id: string) => void
  /** Requests to keep in flight for a batch. 1 = one at a time. */
  parallel: number
  onParallel: (n: number) => void
}) {
  const selected = batch.rows.filter((r) => r.selected)
  const done = selected.filter((r) => r.status).length
  const passed = selected.filter((r) => r.status === 'PASS').length
  const spent = batchCost(selected)
  const allOn = selected.length === batch.rows.length
  const notRun = batch.rows.filter((r) => r.was === null)
  /** Everything the model did not pass: FAIL, and ERROR/PARTIAL/STOPPED too. */
  const notPassed = batch.rows.filter((r) => r.was !== null && r.was !== 'PASS')
  const settledRows = batch.rows.filter((r) => settled(r.was))
  const groups = groupRows(batch.rows, prompts, categories)
  const index = new Map(batch.rows.map((r, i) => [r.id, i + 1]))
  return (
    <div className="batch">
      <header>
        <h1>Batch: {batch.label}</h1>
        <div className="meta">
          <span>
            {selected.length}/{batch.rows.length} selected
          </span>
          <span>{settledRows.length} already run on this model</span>
          {batch.rows.length - settledRows.length > 0 && <span>{batch.rows.length - settledRows.length} not settled</span>}
          {parallel > 1 && <span>{parallel} requests at a time</span>}
          {batch.started && (
            <span>
              {done}/{selected.length} done, {passed} passed
            </span>
          )}
          {spent != null && <span title="Runs with no known price are not counted">{usd(spent)} spent</span>}
        </div>
      </header>
      <div className="runbar">
        {running ? (
          <>
            <button className="btn" onClick={onStop}>
              Stop now
            </button>
            <button className="btn quiet" onClick={onStopAfter}>
              Stop after this one
            </button>
          </>
        ) : (
          <>
            <button className="btn primary" onClick={onStart} disabled={!ready || selected.length === 0}>
              {batch.started ? `Run ${selected.length} again` : `Run ${selected.length} selected`}
            </button>
            <label
              className="param"
              title="How many prompts this batch sends at once. The model server must have spare slots (llama.cpp --parallel / -np) for this to actually go faster; with one slot the requests just queue, and t/s stops being comparable with a sequential run."
            >
              parallel
              <input
                type="number"
                min={1}
                max={MAX_PARALLEL}
                step={1}
                value={parallel}
                aria-label="Parallel requests"
                onChange={(e) => onParallel(clampParallel(e.target.value))}
              />
            </label>
            <button className="btn quiet" onClick={() => onSelect(batch.rows.map((r) => r.id), !allOn)}>
              {allOn ? 'Select none' : 'Select all'}
            </button>
            <button
              className="btn quiet"
              title="Tick only prompts this model has never run"
              onClick={() => {
                onSelect(batch.rows.map((r) => r.id), false)
                onSelect(notRun.map((r) => r.id), true)
              }}
            >
              Select not run ({notRun.length})
            </button>
            <button
              className="btn quiet"
              title="Tick everything this model did not pass, including errors and stops"
              onClick={() => {
                onSelect(batch.rows.map((r) => r.id), false)
                onSelect(notPassed.map((r) => r.id), true)
              }}
            >
              Select failed ({notPassed.length})
            </button>
            <span className="spacer" />
            <button className="btn quiet" onClick={onClose}>
              Close
            </button>
          </>
        )}
      </div>
      {batch.started && (
        <div className="progress" role="progressbar" aria-valuenow={done} aria-valuemax={selected.length}>
          <i style={{ width: `${selected.length ? (100 * done) / selected.length : 0}%` }} />
        </div>
      )}
      {groups.map((g) => {
        const key = g.cat?.slug ?? '_'
        const label = g.cat ? (g.cat.labels[locale] ?? g.cat.labels.en) : 'Other'
        const on = g.rows.filter((r) => r.selected).length
        const state = on === 0 ? 'none' : on === g.rows.length ? 'all' : 'some'
        return (
          <div key={key} role="group" aria-label={label}>
            {groups.length > 1 && (
              <label className="group">
                <input
                  type="checkbox"
                  checked={state === 'all'}
                  ref={(el) => {
                    if (el) el.indeterminate = state === 'some'
                  }}
                  disabled={running}
                  onChange={() => onSelect(g.rows.map((r) => r.id), state !== 'all')}
                  aria-label={`Include group ${label}`}
                />
                <span>{label}</span>
                <span className="n">
                  {on}/{g.rows.length}
                </span>
              </label>
            )}
            {g.rows.map((r) => {
              const p = prompts.find((x) => x.id === r.id)
              const isCurrent = batch.active.includes(r.id)
              return (
                <div key={r.id} className={`row${isCurrent ? ' current' : ''}${r.selected ? '' : ' off'}`}>
                  <label className="pick">
                    <input type="checkbox" checked={r.selected} disabled={running} onChange={() => onSelect([r.id], !r.selected)} aria-label={`Include ${r.id}`} />
                    <span className="n">{index.get(r.id)}</span>
                  </label>
                  {r.status ? (
                    <span className={`status ${r.status}`}>{r.status}</span>
                  ) : (
                    <span className="status NONE">{isCurrent ? 'running' : batch.started && r.selected ? 'queued' : ''}</span>
                  )}
                  <button className="link" onClick={() => onOpen(r.id)} title="Open this prompt on the bench">
                    {p ? promptName(p, locale) : r.id}
                    <span className="n"> {r.id}</span>
                  </button>
                  <span className="n">
                    {r.tps != null ? `${r.tps.toFixed(0)} t/s` : ''}
                    {r.cost != null ? ` · ${usd(r.cost)}` : ''}
                  </span>
                  <span className="n" title={r.detail}>
                    {r.detail.slice(0, 60)}
                  </span>
                  <span className={`was ${r.was ?? ''}`} title={r.was ? 'Last verdict on this model' : 'Never run on this model'}>
                    {r.was ? `was ${r.was}` : 'new'}
                  </span>
                </div>
              )
            })}
          </div>
        )
      })}
    </div>
  )
}
