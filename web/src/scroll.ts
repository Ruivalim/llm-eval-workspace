/** Scroll helpers for the streaming output. Kept out of the component file so
 *  Fast Refresh keeps working. */

/** True when the box is pinned to its bottom, within a small slack. */
export function isAtBottom(
  box: { scrollHeight: number; scrollTop: number; clientHeight: number },
  slack = 48,
) {
  return box.scrollHeight - box.scrollTop - box.clientHeight < slack
}

/** Nearest scrollable ancestor, so following works wherever Output is mounted. */
export function scrollBox(node: HTMLElement | null): HTMLElement | null {
  for (let el = node?.parentElement ?? null; el; el = el.parentElement) {
    if (/auto|scroll/.test(getComputedStyle(el).overflowY)) return el
  }
  return null
}

export type Follower = {
  /** Call after new content lands; scrolls to the bottom only while stuck. */
  tick: () => void
  /** Stick again and jump to the bottom (a fresh run). */
  pin: () => void
  /** Whether the next tick will scroll. */
  stuck: () => boolean
  dispose: () => void
}

/**
 * Keeps a box scrolled to its bottom while content streams in, and stops the
 * moment the reader scrolls up by any amount.
 *
 * The rule is direction, not position: every scroll event that moves up is the
 * reader's (our own scrolls only ever go down), so it releases the follow, even
 * a one-pixel trackpad nudge that a position slack would swallow. Scrolling back
 * down to the bottom sticks again. A shrinking scrollHeight (a collapsing panel)
 * can clamp scrollTop upwards; that is not the reader, so it is ignored.
 */
export function followTail(box: HTMLElement): Follower {
  let stick = true
  let prevTop = box.scrollTop
  let prevHeight = box.scrollHeight

  const onScroll = () => {
    const top = box.scrollTop
    const height = box.scrollHeight
    const movedUp = top < prevTop && height >= prevHeight
    if (movedUp) stick = false
    else if (top > prevTop && isAtBottom(box)) stick = true
    prevTop = top
    prevHeight = height
  }
  box.addEventListener('scroll', onScroll, { passive: true })

  const toBottom = () => {
    box.scrollTop = box.scrollHeight
    prevTop = box.scrollTop
    prevHeight = box.scrollHeight
  }

  return {
    tick: () => {
      if (stick) toBottom()
    },
    pin: () => {
      stick = true
      toBottom()
    },
    stuck: () => stick,
    dispose: () => box.removeEventListener('scroll', onScroll),
  }
}
