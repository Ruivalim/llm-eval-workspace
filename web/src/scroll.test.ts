import { describe, expect, it } from 'vitest'
import { followTail, isAtBottom } from './scroll'

/** The rule that decides whether a stream keeps following the tail. jsdom has no
 *  layout, so this checks the arithmetic directly. */
describe('isAtBottom', () => {
  const box = (scrollTop: number) => ({ scrollHeight: 900, clientHeight: 400, scrollTop })

  it('is true when pinned to the bottom', () => {
    expect(isAtBottom(box(500))).toBe(true)
  })

  it('is false once the reader scrolls up', () => {
    expect(isAtBottom(box(300))).toBe(false)
  })

  it('tolerates a few pixels of slack', () => {
    expect(isAtBottom(box(470))).toBe(true)
    expect(isAtBottom(box(440))).toBe(false)
    expect(isAtBottom(box(440), 100)).toBe(true)
  })

  it('is true when the content does not overflow yet', () => {
    expect(isAtBottom({ scrollHeight: 400, clientHeight: 400, scrollTop: 0 })).toBe(true)
  })
})

/** jsdom does not lay out, so the box is a real element with scroll metrics
 *  stubbed. Setting scrollTop clamps to the content like a browser and fires a
 *  scroll event, which is what followTail listens to. */
function fakeBox(clientHeight = 400) {
  const el = document.createElement('div')
  let scrollHeight = clientHeight
  let scrollTop = 0
  Object.defineProperty(el, 'clientHeight', { get: () => clientHeight })
  Object.defineProperty(el, 'scrollHeight', { get: () => scrollHeight })
  Object.defineProperty(el, 'scrollTop', {
    get: () => scrollTop,
    set: (v: number) => {
      const next = Math.max(0, Math.min(v, scrollHeight - clientHeight))
      if (next === scrollTop) return
      scrollTop = next
      el.dispatchEvent(new Event('scroll'))
    },
  })
  return {
    el,
    /** Content grew by `px`: the streamed delta. */
    grow(px: number) {
      scrollHeight += px
    },
    /** Content shrank (a panel collapsed); the browser clamps scrollTop. */
    shrink(px: number) {
      scrollHeight -= px
      el.scrollTop = scrollTop
    },
    /** The reader moved the wheel. */
    wheel(px: number) {
      el.scrollTop = scrollTop + px
    },
    top: () => scrollTop,
    bottom: () => scrollHeight - clientHeight,
  }
}

describe('followTail', () => {
  it('keeps the bottom in view while content streams', () => {
    const b = fakeBox()
    const f = followTail(b.el)
    for (let i = 0; i < 5; i++) {
      b.grow(120)
      f.tick()
    }
    expect(b.top()).toBe(b.bottom())
    f.dispose()
  })

  it('lets go on a one-pixel scroll up, even within the slack', () => {
    const b = fakeBox()
    const f = followTail(b.el)
    b.grow(600)
    f.tick()
    b.wheel(-1)
    const where = b.top()
    b.grow(120)
    f.tick()
    expect(f.stuck()).toBe(false)
    expect(b.top()).toBe(where)
    f.dispose()
  })

  it('does not get yanked back by a burst of deltas after scrolling up', () => {
    const b = fakeBox()
    const f = followTail(b.el)
    b.grow(600)
    f.tick()
    b.wheel(-200)
    const where = b.top()
    for (let i = 0; i < 50; i++) {
      b.grow(3)
      f.tick()
    }
    expect(b.top()).toBe(where)
    f.dispose()
  })

  it('sticks again when the reader scrolls back to the bottom', () => {
    const b = fakeBox()
    const f = followTail(b.el)
    b.grow(600)
    f.tick()
    b.wheel(-200)
    b.wheel(500)
    expect(f.stuck()).toBe(true)
    b.grow(120)
    f.tick()
    expect(b.top()).toBe(b.bottom())
    f.dispose()
  })

  it('stays released when scrolling down but not all the way', () => {
    const b = fakeBox()
    const f = followTail(b.el)
    b.grow(1000)
    f.tick()
    b.wheel(-500)
    b.wheel(100)
    expect(f.stuck()).toBe(false)
    f.dispose()
  })

  it('ignores the clamp caused by a collapsing panel', () => {
    const b = fakeBox()
    const f = followTail(b.el)
    b.grow(600)
    f.tick()
    b.shrink(300)
    expect(f.stuck()).toBe(true)
    b.grow(200)
    f.tick()
    expect(b.top()).toBe(b.bottom())
    f.dispose()
  })

  it('pin sticks again and jumps to the bottom for a fresh run', () => {
    const b = fakeBox()
    const f = followTail(b.el)
    b.grow(600)
    f.tick()
    b.wheel(-300)
    expect(f.stuck()).toBe(false)
    f.pin()
    expect(f.stuck()).toBe(true)
    expect(b.top()).toBe(b.bottom())
    f.dispose()
  })

  it('stops listening after dispose', () => {
    const b = fakeBox()
    const f = followTail(b.el)
    b.grow(600)
    f.tick()
    f.dispose()
    b.wheel(-100)
    expect(f.stuck()).toBe(true)
  })
})
