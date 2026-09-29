import { useMemo, useState } from 'react'
import type { Category, MatrixCell, Prompt } from '../api'
import { promptName } from '../format'
import { loadClosed, saveClosed } from '../store'

type Props = {
  prompts: Prompt[]
  categories: Category[]
  locale: string
  current: string | null
  model: string | null
  matrix: MatrixCell[]
  busy: boolean
  onPick: (id: string) => void
  onBatch: (ids: string[], label: string) => void
}

const TYPE: Record<string, string> = { text: 'text', render_html: 'html', tool_call: 'tools', error_test: 'fault' }
export function Library({ prompts, categories, locale, current, model, matrix, busy, onPick, onBatch }: Props) {
  const [q, setQ] = useState('')
  const [closed, setClosed] = useState<Set<string>>(loadClosed)
  const last = useMemo(() => {
    const m = new Map<string, MatrixCell>()
    for (const c of matrix) if (c.model === model) m.set(c.prompt_id, c)
    return m
  }, [matrix, model])
  const needle = q.trim().toLowerCase()
  const visible = (p: Prompt) =>
    !needle || p.id.includes(needle) || promptName(p, locale).toLowerCase().includes(needle) || p.tags.some((t) => t.includes(needle))
  // Category order first, then file order inside it: the same order the sidebar shows.
  const ordered = categories.flatMap((cat) => prompts.filter((p) => p.category === cat.slug))
  const auto = ordered.filter((p) => p.auto)

  function toggle(slug: string, open: boolean) {
    setClosed((prev) => {
      const next = new Set(prev)
      if (open) next.delete(slug)
      else next.add(slug)
      saveClosed(next)
      return next
    })
  }

  return (
    <aside className="library" aria-label="Prompt library">
      <input className="search" type="search" placeholder="Filter prompts" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Filter prompts" />
      <div className="groups">
        {categories.map((cat) => {
          const items = prompts.filter((p) => p.category === cat.slug && visible(p))
          if (!items.length) return null
          // A search opens everything it matched; otherwise the user's fold state wins.
          const open = !!needle || !closed.has(cat.slug)
          const label = cat.labels[locale] ?? cat.labels.en
          return (
            <details key={cat.slug} open={open} onToggle={(e) => !needle && toggle(cat.slug, e.currentTarget.open)}>
              <summary>
                <span className="chev" aria-hidden="true" />
                <span className="label" title={label}>{label}</span>
                <span className="count">{items.length}</span>
                <button
                  className="btn quiet"
                  disabled={busy}
                  title="Plan a batch of this category. Prompts this model already ran start unticked, so this runs the new ones."
                  onClick={(e) => {
                    e.preventDefault()
                    onBatch(items.map((p) => p.id), cat.labels.en)
                  }}
                >
                  Run new
                </button>
              </summary>
              {items.map((p) => {
                const cell = last.get(p.id)
                return (
                  <button key={p.id} className="item" aria-current={current === p.id ? 'true' : undefined} onClick={() => onPick(p.id)} disabled={busy}>
                    <span className={`dot ${cell?.status ?? ''}`} title={cell ? `${cell.status} on ${model}` : 'not run on this model'} />
                    <span className="name" title={promptName(p, locale)}>
                      {promptName(p, locale)}
                      {!p.locales[locale] && <span className="type"> ·en</span>}
                    </span>
                    <span className="type">{TYPE[p.eval_type]}</span>
                  </button>
                )
              })}
            </details>
          )
        })}
      </div>
      <div className="foot">
        <button className="btn" disabled={busy || !auto.length} onClick={() => onBatch(auto.map((p) => p.id), 'all auto-checked')}>
          Run all auto-checked ({auto.length})
        </button>
      </div>
    </aside>
  )
}
