import { describe, expect, it } from 'vitest'
import { clampParallel, MAX_PARALLEL, runPool, settled } from './batch'

const tick = () => new Promise((r) => setTimeout(r, 1))

describe('runPool', () => {
  it('keeps at most n in flight and visits every id exactly once', async () => {
    const ids = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    let inFlight = 0
    let peak = 0
    const done: string[] = []
    await runPool(ids, 3, async (id) => {
      inFlight++
      peak = Math.max(peak, inFlight)
      await tick()
      inFlight--
      done.push(id)
    })
    expect(peak).toBe(3)
    expect([...done].sort()).toEqual(ids)
  })

  it('never runs more than there is work for', async () => {
    let inFlight = 0
    let peak = 0
    await runPool(['a', 'b'], 8, async () => {
      inFlight++
      peak = Math.max(peak, inFlight)
      await tick()
      inFlight--
    })
    expect(peak).toBe(2)
  })

  it('finishes the batch when one row throws, then surfaces the error', async () => {
    const done: string[] = []
    const err = runPool(['a', 'b', 'c', 'd'], 2, async (id) => {
      await tick()
      if (id === 'b') throw new Error('boom')
      done.push(id)
    })
    await expect(err).rejects.toThrow('boom')
    expect([...done].sort()).toEqual(['a', 'c', 'd'])
  })

  it('starts nothing new once stop() is true, but lets the in-flight ones land', async () => {
    const started: string[] = []
    const done: string[] = []
    let stopped = false
    let release: () => void = () => {}
    const gate = new Promise<void>((r) => (release = r))
    const run = runPool(
      ['a', 'b', 'c', 'd'],
      2,
      async (id) => {
        started.push(id)
        await gate
        done.push(id)
      },
      () => stopped,
    )
    // Both workers are parked inside their first id here, nothing has finished yet.
    expect(started).toEqual(['a', 'b'])
    stopped = true // like a Stop click landing mid-batch
    release()
    await run
    expect(started).toEqual(['a', 'b'])
    expect([...done].sort()).toEqual(['a', 'b'])
  })
})

describe('clampParallel', () => {
  it('sends anything unusable back to 1, and caps at MAX_PARALLEL', () => {
    expect(clampParallel('')).toBe(1)
    expect(clampParallel('nope')).toBe(1)
    expect(clampParallel(0)).toBe(1)
    expect(clampParallel(-3)).toBe(1)
    expect(clampParallel('2.7')).toBe(2)
    expect(clampParallel(4)).toBe(4)
    expect(clampParallel(1e9)).toBe(MAX_PARALLEL)
  })
})

describe('settled', () => {
  it('counts only a real PASS or FAIL as done', () => {
    expect(settled('PASS')).toBe(true)
    expect(settled('FAIL')).toBe(true)
    expect(settled('ERROR')).toBe(false)
    expect(settled('PARTIAL')).toBe(false)
    expect(settled('PENDING')).toBe(false)
    expect(settled(null)).toBe(false)
  })
})
