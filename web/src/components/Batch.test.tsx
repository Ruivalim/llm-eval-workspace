import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { Category, MatrixCell, Prompt } from '../api'
import { groupRows, planRows } from '../batch'
import { BatchView } from './Batch'

const cats: Category[] = [
  { slug: 'logic', icon: '', labels: { en: 'Logic' } },
  { slug: 'coding', icon: '', labels: { en: 'Coding' } },
]
const prompt = (id: string, category: string): Prompt => ({
  id, category, eval_type: 'text', max_tokens: null, tags: [], tools: [], fault: null,
  check: { type: 'none', params: {} }, locales: { en: { name: id.toUpperCase(), system_prompt: '', prompt: 'x' } }, auto: true,
})
const prompts = [prompt('a', 'logic'), prompt('b', 'logic'), prompt('c', 'coding')]
const cell = (prompt_id: string, model: string, status: MatrixCell['status'], effort: MatrixCell['effort'] = 'default'): MatrixCell => ({
  id: 1, prompt_id, status, tps: 1, ttft_ms: 1, detail: '', model, effort, locale: 'en', server_id: 1, human_score: null,
})

describe('planRows', () => {
  it('only counts a prompt as settled at the effort being planned', () => {
    const matrix = [cell('a', 'm', 'PASS', 'high'), cell('b', 'm', 'PASS', 'off')]
    const atOff = planRows(['a', 'b'], matrix, 'm', 'off')
    expect(atOff.map((r) => [r.id, r.selected, r.was])).toEqual([
      ['a', true, null],
      ['b', false, 'PASS'],
    ])
    expect(planRows(['a', 'b'], matrix, 'm').every((r) => r.selected && r.was === null)).toBe(true)
  })

  it('unticks prompts the model settled, and brings errors back ticked', () => {
    const rows = planRows(
      ['a', 'b', 'c', 'd'],
      [cell('a', 'm', 'PASS'), cell('b', 'm', 'FAIL'), cell('c', 'm', 'ERROR'), cell('d', 'other', 'PASS')],
      'm',
    )
    expect(rows.map((r) => [r.id, r.selected, r.was])).toEqual([
      ['a', false, 'PASS'],
      ['b', false, 'FAIL'],
      ['c', true, 'ERROR'],
      ['d', true, null],
    ])
  })
  it('ticks everything when no model is chosen', () => {
    expect(planRows(['a'], [cell('a', 'm', 'PASS')], null).every((r) => r.selected)).toBe(true)
  })
})

describe('groupRows', () => {
  it('follows category order and parks unknown ids at the end', () => {
    const rows = planRows(['c', 'zzz', 'a'], [], null)
    const g = groupRows(rows, prompts, cats)
    expect(g.map((x) => [x.cat?.slug ?? null, x.rows.map((r) => r.id)])).toEqual([
      ['logic', ['a']],
      ['coding', ['c']],
      [null, ['zzz']],
    ])
  })
})

describe('BatchView', () => {
  afterEach(cleanup)
  const onParallel = vi.fn()
  function mount(rows = planRows(['a', 'b', 'c'], [cell('b', 'm', 'PASS')], 'm'), parallel = 1) {
    const onSelect = vi.fn()
    render(
      <BatchView
        batch={{ label: 't', rows, active: [], started: false }}
        prompts={prompts}
        categories={cats}
        locale="en"
        running={false}
        ready
        onSelect={onSelect}
        onStart={() => {}}
        onStop={() => {}}
        onStopAfter={() => {}}
        onClose={() => {}}
        onOpen={() => {}}
        parallel={parallel}
        onParallel={onParallel}
      />,
    )
    return onSelect
  }

  it('group checkbox is indeterminate when only part of the group is ticked, and ticks the rest', () => {
    const onSelect = mount()
    const logic = screen.getByLabelText('Include group Logic') as HTMLInputElement
    expect(logic.checked).toBe(false)
    expect(logic.indeterminate).toBe(true)
    fireEvent.click(logic)
    expect(onSelect).toHaveBeenCalledWith(['a', 'b'], true)
  })

  it('a fully ticked group unticks as a whole', () => {
    const onSelect = mount(planRows(['a', 'b', 'c'], [], 'm'))
    const logic = screen.getByLabelText('Include group Logic') as HTMLInputElement
    expect(logic.checked).toBe(true)
    fireEvent.click(logic)
    expect(onSelect).toHaveBeenCalledWith(['a', 'b'], false)
  })

  it('shows the previous verdict and offers to select only what never ran', () => {
    const onSelect = mount()
    expect(screen.getByText('was PASS')).toBeTruthy()
    expect(screen.getByText('1 already run on this model')).toBeTruthy()
    fireEvent.click(screen.getByText('Select not run (2)'))
    expect(onSelect).toHaveBeenNthCalledWith(1, ['a', 'b', 'c'], false)
    expect(onSelect).toHaveBeenNthCalledWith(2, ['a', 'c'], true)
  })

  it('offers Select failed for everything that did not pass', () => {
    const onSelect = mount(planRows(['a', 'b', 'c'], [cell('a', 'm', 'FAIL'), cell('b', 'm', 'PASS'), cell('c', 'm', 'ERROR')], 'm'))
    expect(screen.getByText('Select failed (2)')).toBeTruthy()
    fireEvent.click(screen.getByText('Select failed (2)'))
    expect(onSelect).toHaveBeenNthCalledWith(1, ['a', 'b', 'c'], false)
    expect(onSelect).toHaveBeenNthCalledWith(2, ['a', 'c'], true)
  })

  it('offers a parallel count and clamps what the input hands over', () => {
    mount()
    const input = screen.getByLabelText('Parallel requests') as HTMLInputElement
    expect(input.value).toBe('1')
    fireEvent.change(input, { target: { value: '4' } })
    expect(onParallel).toHaveBeenLastCalledWith(4)
    fireEvent.change(input, { target: { value: '' } })
    expect(onParallel).toHaveBeenLastCalledWith(1)
    fireEvent.change(input, { target: { value: '900' } })
    expect(onParallel).toHaveBeenLastCalledWith(16)
  })

  it('says how many requests are in flight only when that is more than one', () => {
    mount(planRows(['a', 'b', 'c'], [], 'm'), 4)
    expect(screen.getByText('4 requests at a time')).toBeTruthy()
  })
})
