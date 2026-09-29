import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { Model, Prices } from '../api'
import type { Aliases } from '../format'
import { builtinFor } from '../prices'
import { Settings } from './Settings'

const fetchMock = vi.fn()
let priceList: Prices

function json(body: unknown, status = 200) {
  return Promise.resolve(new Response(status === 204 ? null : JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } }))
}

beforeEach(() => {
  priceList = { prices: [], builtin: [{ model: 'claude-haiku-4-5', input_per_mtok: 1, output_per_mtok: 5 }], builtin_as_of: '2026-09-25' }
  fetchMock.mockImplementation((url: string, init?: RequestInit) => {
    if (url === '/api/prices' && !init?.method) return json(priceList)
    if (url === '/api/prices' && init?.method === 'PUT') {
      const body = JSON.parse(String(init.body))
      priceList = { ...priceList, prices: [body] }
      return json({ model: body.model, filled: 2 })
    }
    if (url.startsWith('/api/prices?') && init?.method === 'DELETE') {
      priceList = { ...priceList, prices: [] }
      return json(null, 204)
    }
    return json({ detail: 'unexpected' }, 500)
  })
  vi.stubGlobal('fetch', fetchMock)
})

afterEach(() => {
  cleanup()
  fetchMock.mockReset()
  vi.unstubAllGlobals()
})

const models: Model[] = [{ id: 'org/Model-GGUF', loaded: false }]

function mount(aliases: Aliases = {}, onChange = vi.fn(), list: Model[] = models) {
  render(
    <Settings
      servers={[]}
      onServersChange={() => {}}
      settings={{ global_system_prompt: '' }}
      onSettingsChange={() => {}}
      models={list}
      aliases={aliases}
      onAliasesChange={onChange}
    />,
  )
  return onChange
}

describe('Model labels', () => {
  it('offers the automatic label as the placeholder, so empty means automatic', () => {
    mount()
    const input = screen.getByLabelText('Label for org/Model-GGUF') as HTMLInputElement
    expect(input.value).toBe('')
    expect(input.placeholder).toBe('Model')
  })

  it('writes a typed alias and drops it again when cleared', () => {
    const onChange = mount()
    const input = screen.getByLabelText('Label for org/Model-GGUF')
    fireEvent.change(input, { target: { value: 'ornith-35b' } })
    expect(onChange).toHaveBeenLastCalledWith({ 'org/Model-GGUF': 'ornith-35b' })

    cleanup()
    const onChange2 = mount({ 'org/Model-GGUF': 'ornith-35b' })
    fireEvent.change(screen.getByLabelText('Label for org/Model-GGUF'), { target: { value: '  ' } })
    expect(onChange2).toHaveBeenLastCalledWith({})
  })
})

describe('Prices', () => {
  it('lists the models of the selected server, with the list price as a hint for Claude ids', async () => {
    mount({}, vi.fn(), [{ id: 'deepseek-chat', loaded: false }, { id: 'claude-haiku-4-5-20251001', loaded: false }])
    const deepIn = (await screen.findByLabelText('Input price for deepseek-chat')) as HTMLInputElement
    expect(deepIn.value).toBe('')
    expect(deepIn.placeholder).toBe('no price')
    await waitFor(() => expect((screen.getByLabelText('Output price for claude-haiku-4-5-20251001') as HTMLInputElement).placeholder).toBe('5'))
  })

  it('saves a price, accepts a decimal comma, and says how many old results got a cost', async () => {
    mount({}, vi.fn(), [{ id: 'deepseek-chat', loaded: false }])
    const save = await screen.findByLabelText('Save price for deepseek-chat')
    expect((save as HTMLButtonElement).disabled).toBe(true)
    fireEvent.change(screen.getByLabelText('Input price for deepseek-chat'), { target: { value: '0,27' } })
    expect((save as HTMLButtonElement).disabled).toBe(true) // output still empty
    fireEvent.change(screen.getByLabelText('Output price for deepseek-chat'), { target: { value: '1.1' } })
    fireEvent.click(save)
    await screen.findByText(/2 earlier results got a cost/)
    const put = fetchMock.mock.calls.find(([, init]) => init?.method === 'PUT')!
    expect(JSON.parse(String(put[1].body))).toEqual({ model: 'deepseek-chat', input_per_mtok: 0.27, output_per_mtok: 1.1 })
    // the saved price comes back into the fields, and Remove appears
    await screen.findByLabelText('Remove price for deepseek-chat')
    expect((screen.getByLabelText('Input price for deepseek-chat') as HTMLInputElement).value).toBe('0.27')
  })

  it('rejects negative and non-numeric prices before they reach the API', async () => {
    mount({}, vi.fn(), [{ id: 'm', loaded: false }])
    const save = (await screen.findByLabelText('Save price for m')) as HTMLButtonElement
    fireEvent.change(screen.getByLabelText('Input price for m'), { target: { value: '-1' } })
    fireEvent.change(screen.getByLabelText('Output price for m'), { target: { value: '2' } })
    expect(save.disabled).toBe(true)
    fireEvent.change(screen.getByLabelText('Input price for m'), { target: { value: 'abc' } })
    expect(save.disabled).toBe(true)
  })

  it('keeps showing a priced model after the header switches to another server', async () => {
    priceList.prices = [{ model: 'gone-model', input_per_mtok: 1, output_per_mtok: 2 }]
    mount({}, vi.fn(), [])
    expect(((await screen.findByLabelText('Input price for gone-model')) as HTMLInputElement).value).toBe('1')
    fireEvent.click(screen.getByLabelText('Remove price for gone-model'))
    await screen.findByText(/Costs already recorded stay/)
  })

  it('matches a dated Claude snapshot to its list price, and nothing looser', () => {
    const builtin = [
      { model: 'claude-opus-5', input_per_mtok: 5, output_per_mtok: 25 },
      { model: 'claude-opus-5-5', input_per_mtok: 4, output_per_mtok: 20 },
    ]
    expect(builtinFor('claude-opus-5-5', builtin)?.input_per_mtok).toBe(4)
    expect(builtinFor('claude-opus-5-20260101', builtin)?.input_per_mtok).toBe(5)
    expect(builtinFor('claude-opus-5-5-preview', builtin)).toBeUndefined()
  })
})

describe('Server form', () => {
  it('fills the Anthropic URL when that flavor is picked on an empty form', async () => {
    mount()
    const url = screen.getByPlaceholderText('http://localhost:11434/v1') as HTMLInputElement
    fireEvent.change(screen.getByDisplayValue('openai'), { target: { value: 'anthropic' } })
    expect(url.value).toBe('https://api.anthropic.com')
  })
})
