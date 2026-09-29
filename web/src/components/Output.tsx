import { useCallback, useEffect, useMemo, useRef } from 'react'
import type { Delta, EvalType } from '../api'
import { renderMarkdown } from '../markdown'
import { followTail, scrollBox, type Follower } from '../scroll'

type Props = { evalType: EvalType; deltas: Delta[]; running: boolean; artifactSrcDoc: string | null }

function joined(deltas: Delta[], kind: Delta['kind']) {
  let s = ''
  for (const d of deltas) if (d.kind === kind) s += d.text
  return s
}

/**
 * Follows the tail of a scroll box while streaming, and lets go the moment the
 * reader scrolls up. Calling scrollIntoView on every delta used to yank the view
 * back to the bottom, which made reading (or even scrolling) mid-stream impossible.
 *
 * `find` resolves the box lazily because Output renders null until the first
 * delta, so the element may not exist when the hook first runs.
 */
function useFollowTail(find: () => HTMLElement | null, running: boolean, tick: number) {
  const follower = useRef<{ box: HTMLElement; f: Follower } | null>(null)

  useEffect(() => {
    const box = find()
    if (follower.current && follower.current.box !== box) {
      follower.current.f.dispose()
      follower.current = null
    }
    if (!box) return
    if (!follower.current) follower.current = { box, f: followTail(box) }
    if (running) follower.current.f.tick()
  }, [find, running, tick])

  // A fresh run starts pinned again.
  useEffect(() => {
    if (running) follower.current?.f.pin()
  }, [running])

  useEffect(() => () => follower.current?.f.dispose(), [])
}

export function Output({ evalType, deltas, running, artifactSrcDoc }: Props) {
  const answer = useMemo(() => joined(deltas, 'content'), [deltas])
  const reasoning = useMemo(() => joined(deltas, 'reasoning'), [deltas])
  const endRef = useRef<HTMLDivElement>(null)
  useFollowTail(useCallback(() => scrollBox(endRef.current), []), running, deltas.length)

  if (!deltas.length && !running) return null

  if (evalType === 'render_html') {
    return (
      <div className="output">
        {reasoning && <Reasoning text={reasoning} open={running} />}
        <div className="split">
          <pre className={`code${running ? ' cursor' : ''}`}>{running ? answer.slice(-6000) : answer}</pre>
          {artifactSrcDoc ? (
            <iframe title="Rendered answer" sandbox="allow-scripts" srcDoc={artifactSrcDoc} />
          ) : (
            <div className="waiting">{running ? 'Renders when the stream ends' : 'Nothing to render'}</div>
          )}
        </div>
        <div ref={endRef} />
      </div>
    )
  }

  if (evalType === 'tool_call') {
    return (
      <div className="output">
        <div className="trace" aria-live="polite">
          {deltas
            .filter((d) => d.kind !== 'content' && d.kind !== 'reasoning')
            .map((d, i) => (
              <div key={i} className={`l ${d.kind}`}>
                <span className="k">{d.kind === 'tool_call' ? 'call' : 'result'}</span>
                <span>{d.text}</span>
              </div>
            ))}
          {reasoning && (
            <div className="l reasoning">
              <span className="k">think</span>
              <span>{reasoning}</span>
            </div>
          )}
          {(answer || running) && (
            <div className="l">
              <span className="k">answer</span>
              <span className={running ? 'cursor' : ''}>{answer}</span>
            </div>
          )}
        </div>
        <div ref={endRef} />
      </div>
    )
  }

  return (
    <div className="output">
      {reasoning && <Reasoning text={reasoning} open={running && !answer} />}
      <div className={`answer${running ? ' cursor' : ''}`} aria-live="polite" dangerouslySetInnerHTML={{ __html: renderMarkdown(answer) }} />
      <div ref={endRef} />
    </div>
  )
}

/** The reasoning panel scrolls on its own, so it follows its own tail too. */
function Reasoning({ text, open }: { text: string; open: boolean }) {
  const preRef = useRef<HTMLPreElement>(null)
  useFollowTail(useCallback(() => preRef.current, []), open, text.length)
  return (
    <details className="think" open={open}>
      <summary>Reasoning · {text.length.toLocaleString()} chars</summary>
      <pre ref={preRef}>{text}</pre>
    </details>
  )
}
