import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { HistoryRow, MatrixCell, Prompt, Server } from '../api'
import { Starter } from './Starter'

afterEach(cleanup)

const prompts: Prompt[] = ['p1', 'p2'].map((id) => ({
  id, category: 'logic', eval_type: 'text', max_tokens: null, tags: [], tools: [], fault: null,
  check: { type: 'none', params: {} }, locales: { en: { name: `Name ${id}`, system_prompt: '', prompt: 'x' } }, auto: true,
}))
const cell = (prompt_id: string, status: MatrixCell['status']): MatrixCell => ({
  id: 1, prompt_id, status, tps: 1, ttft_ms: 1, detail: '', model: 'org/Model-GGUF', effort: 'default', locale: 'en', server_id: 1, human_score: null,
})
const server: Server = { id: 1, name: 's', flavor: 'openai', base_url: 'http://x', has_key: false }
const run = (id: number): HistoryRow => ({
  id, run_id: id, prompt_id: 'p1', eval_type: 'text', status: 'PASS', detail: 'ok', artifact: null, trace: false,
  temperature: 0, max_tokens: null, ttft_ms: 1, total_ms: 2, tokens_in: 1, tokens_out: 1, tokens_think: 0, tps: 10,
  created_at: '2026-09-14T12:00:00Z', model: 'org/Model-GGUF', locale: 'en', server_id: 1, server_name: 's', parallel: 1, effort: 'default', human_score: null,
  cost_usd: null, cost_source: '',
})

function mount(over: Partial<Parameters<typeof Starter>[0]> = {}) {
  const onOpenSettings = vi.fn()
  render(
    <Starter
      server={server}
      model="org/Model-GGUF"
      locale="en"
      aliases={{}}
      prompts={prompts}
      matrix={[]}
      recent={[]}
      onOpenSettings={onOpenSettings}
      {...over}
    />,
  )
  return onOpenSettings
}

describe('Starter', () => {
  it('sends a first run to Settings and names a working base URL', () => {
    const onOpenSettings = mount({ server: null, model: null })
    expect(screen.getByText(/No server configured yet/)).toBeTruthy()
    expect(screen.getByText('http://localhost:11434/v1')).toBeTruthy()
    fireEvent.click(screen.getByText('Open Settings'))
    expect(onOpenSettings).toHaveBeenCalledOnce()
  })

  it('asks for a model when a server is there but none is picked', () => {
    mount({ model: null })
    expect(screen.getByText(/No model selected/)).toBeTruthy()
    expect(screen.queryByText('Passed')).toBeNull()
  })

  it('summarises the selected model from the matrix', () => {
    mount({ matrix: [cell('p1', 'PASS'), cell('p2', 'FAIL')] })
    expect(screen.getByText('1 of 2 prompts')).toBeTruthy()
    expect(screen.getByText('2 on this model')).toBeTruthy()
  })

  it('lists the last runs with the short model label and the prompt name', () => {
    mount({ matrix: [], recent: [run(7)], aliases: { 'org/Model-GGUF': 'ornith-35b' } })
    expect(screen.getByText('Last runs')).toBeTruthy()
    const table = within(screen.getByRole('table'))
    // The alias shows up in the stats block too, so scope the assertion to the table.
    expect(table.getByText('ornith-35b')).toBeTruthy()
    expect(table.getByText('Name p1')).toBeTruthy()
    expect(table.getByText('PASS')).toBeTruthy()
  })
})
