/** End-to-end of the batch path inside the React tree: Library "run all
 *  auto-checked" -> plan -> run at parallel 2 -> every row gets a verdict and the
 *  shell is told the run finished. The model side is mocked; nothing else is. */
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import type { Category, Delta, Meta, Outcome, Prompt } from '../api'
import type { Prefs } from '../store'
import { Bench } from './Bench'

const evaluate = vi.fn()
vi.mock('../api', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../api')>()),
  api: { evaluate: (...a: unknown[]) => evaluate(...a), history: vi.fn(async () => []) },
}))

const categories: Category[] = [{ slug: 'logic', icon: '', labels: { en: 'Logic' } }]
const prompt = (id: string): Prompt => ({
  id, category: 'logic', eval_type: 'text', max_tokens: null, tags: [], tools: [], fault: null,
  check: { type: 'none', params: {} }, locales: { en: { name: `Name ${id}`, system_prompt: '', prompt: 'x' } }, auto: true,
})
const meta: Meta = {
  locales: ['en'], default_locale: 'en', categories, prompts: [prompt('a'), prompt('b'), prompt('c')],
}
const prefs: Prefs = {
  serverId: 1, model: 'm', locale: 'en', promptId: null, temperature: 0, topP: 1, timeoutS: 60, effort: 'default', parallel: 2, aliases: {}, view: 'bench',
}
const outcome = (id: number, cost_usd: number | null = null): Outcome => ({
  result_id: id, status: 'PASS', detail: 'ok', error: null, artifact: null, trace: false,
  finish_reason: 'stop', answer: '4', reasoning: '', tool_calls: [],
  telemetry: { ttft_ms: 1, total_ms: 2, tokens_in: 1, tokens_out: 1, tokens_think: 0, tps: 10, usage_from_server: true },
  cost_usd, cost_source: cost_usd == null ? '' : 'price',
})

let inFlight = 0
let peak = 0

afterEach(() => {
  cleanup()
  evaluate.mockReset()
  inFlight = 0
  peak = 0
})

it('runs the batch two at a time and lands a verdict on every row', async () => {
  evaluate.mockImplementation(async (_body: unknown, _onDelta: (d: Delta) => void) => {
    inFlight++
    peak = Math.max(peak, inFlight)
    await new Promise((r) => setTimeout(r, 10))
    inFlight--
    return outcome(1)
  })
  const onRunState = vi.fn()
  render(
    <Bench
      meta={meta}
      server={{ id: 1, name: 's', flavor: 'openai', base_url: 'http://x', has_key: false }}
      prefs={prefs}
      update={() => {}}
      matrix={[]}
      refreshMatrix={() => {}}
      globalSystem=""
      onRunState={onRunState}
    />,
  )

  fireEvent.click(screen.getByText('Run all auto-checked (3)'))
  const parallel = screen.getByLabelText('Parallel requests') as HTMLInputElement
  expect(parallel.value).toBe('2')
  fireEvent.click(screen.getByText('Run 3 selected'))

  await waitFor(() => expect(screen.getAllByText('PASS')).toHaveLength(3))
  expect(peak).toBe(2)
  expect(evaluate).toHaveBeenCalledTimes(3)
  const sent = evaluate.mock.calls.map((c) => c[0] as { prompt_id: string; parallel: number })
  expect(sent.map((b) => b.parallel)).toEqual([2, 2, 2])
  expect(new Set(sent.map((b) => b.prompt_id))).toEqual(new Set(['a', 'b', 'c']))
  expect(onRunState).toHaveBeenLastCalledWith(false, expect.any(Function))
})

it('keeps the running batch when a row prompt is opened, and leads back to it', async () => {
  let release!: () => void
  const gate = new Promise<void>((r) => (release = r))
  evaluate.mockImplementation(async () => {
    await gate
    return outcome(1)
  })
  let current: Prefs = { ...prefs, promptId: null }
  const update = vi.fn((p: Partial<Prefs>) => {
    current = { ...current, ...p }
  })
  const bench = (p: Prefs) => (
    <Bench
      meta={meta}
      server={{ id: 1, name: 's', flavor: 'openai', base_url: 'http://x', has_key: false }}
      prefs={p}
      update={update}
      matrix={[]}
      refreshMatrix={() => {}}
      globalSystem=""
      onRunState={() => {}}
    />
  )
  const { rerender } = render(bench(current))

  fireEvent.click(screen.getByText('Run all auto-checked (3)'))
  fireEvent.click(screen.getByText('Run 3 selected'))
  await waitFor(() => expect(screen.getByText('Stop now')).toBeTruthy())

  // Opening a row mid-run used to wipe the batch view with no way back.
  fireEvent.click(screen.getAllByTitle('Open this prompt on the bench')[0])
  expect(update).toHaveBeenCalledWith({ promptId: 'a' })
  rerender(bench({ ...current, promptId: 'a' }))
  expect(screen.getByText(/Back to batch: all auto-checked/)).toBeTruthy()
  expect(screen.getByText(/running/)).toBeTruthy()

  fireEvent.click(screen.getByText(/Back to batch/))
  expect(update).toHaveBeenCalledWith({ promptId: null })
  rerender(bench({ ...current, promptId: null }))
  expect(screen.getByText('Batch: all auto-checked')).toBeTruthy()
  expect(screen.getByText('Stop now')).toBeTruthy()

  release()
  await waitFor(() => expect(screen.getAllByText('PASS')).toHaveLength(3))
})

it('reports a model error on its row without killing the rest of the batch', async () => {
  let n = 0
  evaluate.mockImplementation(async () => {
    n++
    await new Promise((r) => setTimeout(r, 1))
    if (n === 2) throw new Error('connection failed')
    return outcome(1)
  })
  render(
    <Bench
      meta={meta}
      server={{ id: 1, name: 's', flavor: 'openai', base_url: 'http://x', has_key: false }}
      prefs={prefs}
      update={() => {}}
      matrix={[]}
      refreshMatrix={() => {}}
      globalSystem=""
      onRunState={() => {}}
    />,
  )

  fireEvent.click(screen.getByText('Run all auto-checked (3)'))
  fireEvent.click(screen.getByText('Run 3 selected'))

  await waitFor(() => expect(screen.getAllByText('PASS')).toHaveLength(2))
  expect(screen.getByText('ERROR')).toBeTruthy()
  expect(screen.getByText('connection failed')).toBeTruthy()
})

it('adds up what the batch spent, counting only runs with a known cost', async () => {
  let n = 0
  evaluate.mockImplementation(async () => {
    n++
    // one run on a model nobody priced: unknown, not free
    return outcome(n, n === 3 ? null : 0.0125)
  })
  render(
    <Bench
      meta={meta}
      server={{ id: 1, name: 's', flavor: 'openai', base_url: 'http://x', has_key: false }}
      prefs={{ ...prefs, parallel: 1 }}
      update={() => {}}
      matrix={[]}
      refreshMatrix={() => {}}
      globalSystem=""
      onRunState={() => {}}
    />,
  )
  expect(screen.queryByText(/spent/)).toBeNull()
  fireEvent.click(screen.getByText('Run all auto-checked (3)'))
  fireEvent.click(screen.getByText('Run 3 selected'))
  await waitFor(() => expect(screen.getAllByText('PASS')).toHaveLength(3))
  expect(screen.getByText('$0.03 spent')).toBeTruthy()
  expect(screen.getAllByText(/· \$0\.01/)).toHaveLength(2)
})

it('sends the effort from prefs with every run', async () => {
  evaluate.mockImplementation(async () => outcome(1))
  render(
    <Bench
      meta={meta}
      server={{ id: 1, name: 's', flavor: 'openai', base_url: 'http://x', has_key: false }}
      prefs={{ ...prefs, effort: 'off', parallel: 1 }}
      update={() => {}}
      matrix={[]}
      refreshMatrix={() => {}}
      globalSystem=""
      onRunState={() => {}}
    />,
  )
  fireEvent.click(screen.getByText('Run all auto-checked (3)'))
  fireEvent.click(screen.getByText('Run 3 selected'))
  await waitFor(() => expect(evaluate).toHaveBeenCalledTimes(3))
  expect(evaluate.mock.calls.every((c) => (c[0] as { effort: string }).effort === 'off')).toBe(true)
})
