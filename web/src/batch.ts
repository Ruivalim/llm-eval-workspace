/** Batch planning helpers: pure, so they test without a DOM. */
import type { Category, Effort, MatrixCell, Prompt, Status } from './api'

export type BatchRow = {
  id: string
  selected: boolean
  status: string | null
  tps: number | null
  detail: string
  was: Status | null
  /** USD this run cost; null while unknown (not run yet, or no price for the model). */
  cost: number | null
}
export type Batch = { label: string; rows: BatchRow[]; active: string[]; started: boolean }

/** A previous verdict that counts as done. Only a real PASS or FAIL settles a prompt;
 *  ERROR, PARTIAL, STOPPED and never-run all mean "not settled", so a new plan brings
 *  them back ticked instead of making the user hunt them down row by row. */
export function settled(s: Status | null): boolean {
  return s === 'PASS' || s === 'FAIL'
}

/** Runs `work` over `ids` with at most `n` in flight, pulling the next id as each
 *  one lands. Errors are the caller's business (it catches inside `work`), so one
 *  bad row never stops the rest. `stop` is polled before every new id, which is
 *  what "stop after this one" means when several are in the air. */
export async function runPool(
  ids: string[],
  n: number,
  work: (id: string) => Promise<void>,
  stop: () => boolean = () => false,
): Promise<void> {
  let next = 0
  let failed: unknown = null
  const worker = async () => {
    while (next < ids.length && !stop()) {
      const id = ids[next++]
      try {
        await work(id)
      } catch (e) {
        // Drain the rest before surfacing: one broken row must not strand the batch.
        failed ??= e
      }
    }
  }
  const size = Math.max(1, Math.min(Math.trunc(n) || 1, ids.length))
  await Promise.all(Array.from({ length: size }, worker))
  if (failed) throw failed
}

export const MAX_PARALLEL = 16

/** Clamps whatever a number input hands over ("", 1e9, "2.7") to a usable count. */
export function clampParallel(value: unknown): number {
  const n = Math.trunc(Number(value))
  if (!Number.isFinite(n)) return 1
  return Math.min(MAX_PARALLEL, Math.max(1, n))
}

/** Rows for a new batch. Prompts the model already settled (PASS or FAIL) start unticked, so
 *  "run all" fills the gaps and retries whatever errored, instead of repeating what is done. */
export function planRows(ids: string[], matrix: MatrixCell[], model: string | null, effort: Effort = 'default'): BatchRow[] {
  // Settled means settled at this effort: a PASS at high says nothing about off.
  const was = new Map<string, Status>()
  for (const c of matrix) if (c.model === model && c.effort === effort) was.set(c.prompt_id, c.status)
  return ids.map((id) => {
    const prev = was.get(id) ?? null
    return { id, selected: !settled(prev), status: null, tps: null, detail: '', was: prev, cost: null }
  })
}

export type Group = { cat: Category | null; rows: BatchRow[] }

/** Rows grouped in category order; ids with no known prompt land in a trailing group. */
export function groupRows(rows: BatchRow[], prompts: Prompt[], categories: Category[]): Group[] {
  const byId = new Map(prompts.map((p) => [p.id, p]))
  const groups = new Map<string, Group>()
  for (const cat of categories) groups.set(cat.slug, { cat, rows: [] })
  const orphan: Group = { cat: null, rows: [] }
  for (const r of rows) {
    const g = groups.get(byId.get(r.id)?.category ?? '')
    ;(g ?? orphan).rows.push(r)
  }
  return [...groups.values(), orphan].filter((g) => g.rows.length)
}

/** What the finished rows of a batch cost, or null when none had a known price. */
export function batchCost(rows: BatchRow[]): number | null {
  const known = rows.filter((r) => r.cost != null)
  return known.length ? known.reduce((sum, r) => sum + (r.cost ?? 0), 0) : null
}
