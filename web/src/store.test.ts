import { afterEach, describe, expect, it } from 'vitest'
import { loadPrefs } from './store'

afterEach(() => localStorage.clear())

describe('loadPrefs effort', () => {
  it('starts at default, including for prefs saved before effort existed', () => {
    expect(loadPrefs().effort).toBe('default')
    localStorage.setItem('lew.prefs.v1', JSON.stringify({ model: 'm', temperature: 0.2 }))
    const p = loadPrefs()
    expect(p.effort).toBe('default')
    expect(p.model).toBe('m')
  })
  it('keeps a known level and drops an unknown one', () => {
    localStorage.setItem('lew.prefs.v1', JSON.stringify({ effort: 'off' }))
    expect(loadPrefs().effort).toBe('off')
    localStorage.setItem('lew.prefs.v1', JSON.stringify({ effort: 'turbo' }))
    expect(loadPrefs().effort).toBe('default')
  })
})
