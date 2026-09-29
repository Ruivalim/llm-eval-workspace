import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { Category, Prompt } from '../api'
import { loadClosed } from '../store'
import { Library } from './Library'

const cats: Category[] = [
  { slug: 'logic', icon: '', labels: { en: 'Logic' } },
  { slug: 'coding', icon: '', labels: { en: 'Coding' } },
]
const prompt = (id: string, category: string): Prompt => ({
  id, category, eval_type: 'text', max_tokens: null, tags: [], tools: [], fault: null,
  check: { type: 'none', params: {} }, locales: { en: { name: `Name ${id}`, system_prompt: '', prompt: 'x' } }, auto: true,
})
const prompts = [prompt('a1', 'logic'), prompt('c1', 'coding')]

function mount() {
  const onBatch = vi.fn()
  render(<Library prompts={prompts} categories={cats} locale="en" current={null} model="m" matrix={[]} busy={false} onPick={() => {}} onBatch={onBatch} />)
  return onBatch
}
const details = (label: string) => screen.getByText(label).closest('details') as HTMLDetailsElement

describe('Library accordion', () => {
  beforeEach(() => localStorage.clear())
  afterEach(cleanup)

  it('starts open, folds on click and remembers the fold', async () => {
    mount()
    expect(details('Logic').open).toBe(true)
    fireEvent.click(screen.getByText('Logic'))
    expect(details('Logic').open).toBe(false)
    // The toggle event is queued by the browser, so the fold lands in storage a tick later.
    await waitFor(() => expect([...loadClosed()]).toEqual(['logic']))
  })

  it('restores a folded group from storage', () => {
    localStorage.setItem('lew.library.closed.v1', JSON.stringify(['coding']))
    mount()
    expect(details('Coding').open).toBe(false)
    expect(details('Logic').open).toBe(true)
  })

  it('a search opens folded groups without forgetting the fold', () => {
    localStorage.setItem('lew.library.closed.v1', JSON.stringify(['coding']))
    mount()
    fireEvent.change(screen.getByLabelText('Filter prompts'), { target: { value: 'c1' } })
    expect(details('Coding').open).toBe(true)
    expect(screen.queryByText('Logic')).toBeNull()
    expect([...loadClosed()]).toEqual(['coding'])
  })

  it('ignores garbage in storage', () => {
    localStorage.setItem('lew.library.closed.v1', '{not json')
    expect(loadClosed().size).toBe(0)
  })

  it('the run button on a group plans that group without toggling the fold', async () => {
    const onBatch = mount()
    fireEvent.click(screen.getAllByText('Run new')[0])
    expect(onBatch).toHaveBeenCalledWith(['a1'], 'Logic')
    await waitFor(() => expect(details('Logic').open).toBe(true))
  })
})
