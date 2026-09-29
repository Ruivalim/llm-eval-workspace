import { describe, expect, it } from 'vitest'
import { parseSseBlock, readSse } from './sse'

function stream(chunks: string[]): ReadableStream<Uint8Array> {
  const enc = new TextEncoder()
  return new ReadableStream({
    start(c) {
      for (const ch of chunks) c.enqueue(enc.encode(ch))
      c.close()
    },
  })
}

async function collect(s: ReadableStream<Uint8Array>) {
  const out = []
  for await (const ev of readSse(s)) out.push(ev)
  return out
}

describe('parseSseBlock', () => {
  it('defaults event to message and joins multi-line data', () => {
    expect(parseSseBlock('data: a\ndata: b')).toEqual({ event: 'message', data: 'a\nb' })
  })
  it('reads event name and ignores comments', () => {
    expect(parseSseBlock(': ping\nevent: delta\ndata: {"x":1}')).toEqual({ event: 'delta', data: '{"x":1}' })
  })
  it('returns null without data', () => {
    expect(parseSseBlock('event: x')).toBeNull()
  })
})

describe('readSse', () => {
  it('reassembles events split across chunks', async () => {
    const events = await collect(stream(['event: del', 'ta\ndata: {"t":"h', 'i"}\n\nevent: outcome\ndata: {}\n\n']))
    expect(events).toEqual([
      { event: 'delta', data: '{"t":"hi"}' },
      { event: 'outcome', data: '{}' },
    ])
  })
  it('emits a trailing event without the final blank line', async () => {
    expect(await collect(stream(['data: tail']))).toEqual([{ event: 'message', data: 'tail' }])
  })
  it('handles CRLF', async () => {
    expect(await collect(stream(['data: a\r\n\r\n']))).toEqual([{ event: 'message', data: 'a' }])
  })
})
