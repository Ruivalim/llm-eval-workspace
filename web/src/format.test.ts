import { describe, expect, it } from 'vitest'
import { autoModelLabel, modelLabel, usd, withEffort } from './format'

describe('autoModelLabel', () => {
  it('drops the org and the packaging suffix', () => {
    expect(autoModelLabel('gbuzhf/Ornith-1.5-35B-A3B-Abliterated-MTP-UD-APEX-GGUF')).toBe('Ornith-1.5-35B-A3B-Abliterated-MTP-UD-APEX')
    expect(autoModelLabel('mradermacher/Gemma-The-Writer-9B-HERETIC-Uncensored-Abliterated-GGUF')).toBe(
      'Gemma-The-Writer-9B-HERETIC-Uncensored-Abliterated',
    )
  })

  it('peels more than one suffix off the end', () => {
    expect(autoModelLabel('x/Some-Model-i1-GGUF')).toBe('Some-Model')
    expect(autoModelLabel('x/Some-Model-Q4_K_M.gguf')).toBe('Some-Model')
  })

  it('leaves an id it cannot shorten alone', () => {
    expect(autoModelLabel('ornith-35b')).toBe('ornith-35b')
    expect(autoModelLabel('Qwen3.5-9B')).toBe('Qwen3.5-9B')
  })

  it('never returns an empty label', () => {
    expect(autoModelLabel('org/GGUF')).toBe('GGUF')
    expect(autoModelLabel('')).toBe('')
  })
})

describe('modelLabel', () => {
  it('prefers the alias, and falls back when it is blank or missing', () => {
    expect(modelLabel('org/Model-GGUF', { 'org/Model-GGUF': 'ornith-35b' })).toBe('ornith-35b')
    expect(modelLabel('org/Model-GGUF', { 'org/Model-GGUF': '   ' })).toBe('Model')
    expect(modelLabel('org/Model-GGUF')).toBe('Model')
    expect(modelLabel('org/Model-GGUF', {})).toBe('Model')
  })
})

describe('usd', () => {
  it('keeps unknown apart from free', () => {
    expect(usd(null)).toBe('-')
    expect(usd(undefined)).toBe('-')
    expect(usd(0)).toBe('$0')
  })

  it('shows sub-cent runs with four digits and never rounds a real cost down to zero', () => {
    expect(usd(0.0045)).toBe('$0.0045')
    expect(usd(0.000135)).toBe('$0.0001')
    expect(usd(0.00000002)).toBe('<$0.0001')
  })

  it('uses cents from one cent up', () => {
    expect(usd(0.01)).toBe('$0.01')
    expect(usd(1.2749)).toBe('$1.27')
    expect(usd(24)).toBe('$24.00')
  })
})

describe('withEffort', () => {
  it('adds nothing for default, so old runs read as before', () => {
    expect(withEffort('qwen3', 'default')).toBe('qwen3')
    expect(withEffort('qwen3', undefined)).toBe('qwen3')
  })
  it('names off and levels apart', () => {
    expect(withEffort('mimo', 'off')).toBe('mimo · thinking off')
    expect(withEffort('claude', 'xhigh')).toBe('claude · effort xhigh')
  })
})
