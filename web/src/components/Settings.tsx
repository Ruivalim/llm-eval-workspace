import type React from 'react'
import { useEffect, useState } from 'react'
import { api, type AppSettings, type Model, type ModelPrice, type Prices as PriceList, type Server } from '../api'
import { autoModelLabel, type Aliases } from '../format'
import { builtinFor } from '../prices'

const EMPTY = { name: '', base_url: '', api_key: '', flavor: 'openai' }
const ANTHROPIC_URL = 'https://api.anthropic.com'
const PLACEHOLDER: Record<string, string> = { openai: 'http://localhost:11434/v1', unsloth: 'http://localhost:8888/v1', anthropic: ANTHROPIC_URL }

type Props = {
  servers: Server[]
  onServersChange: () => void
  settings: AppSettings
  onSettingsChange: (s: AppSettings) => void
  /** Models of the server selected in the header, so labels can be set for them here. */
  models: Model[]
  aliases: Aliases
  onAliasesChange: (a: Aliases) => void
}

export function Settings({ servers, onServersChange, settings, onSettingsChange, models, aliases, onAliasesChange }: Props) {
  return (
    <div className="page">
      <h1>Settings</h1>
      <GlobalSystem settings={settings} onChange={onSettingsChange} />
      <ModelLabels models={models} aliases={aliases} onChange={onAliasesChange} />
      <Prices models={models} />
      <Servers servers={servers} onChange={onServersChange} />
    </div>
  )
}

function ModelLabels({ models, aliases, onChange }: { models: Model[]; aliases: Aliases; onChange: (a: Aliases) => void }) {
  /** Empty clears the override, so the automatic label takes over again. */
  function set(id: string, value: string) {
    const next = { ...aliases }
    const v = value.trim()
    if (v) next[id] = v
    else delete next[id]
    onChange(next)
  }
  return (
    <section className="settings-block">
      <h2>Model labels</h2>
      <p className="lede">
        Ids like <code>mradermacher/Qwen3.5-9B-The-Bradbury-F451-Pro-Writer-Uncensored-Heretic-i1-GGUF</code> do not fit in the header, the run
        button or the Results tables. Whatever you type below is what those show. Leave it empty to fall back to the automatic label, which drops
        the org and the packaging suffix.
      </p>
      {models.length === 0 ? (
        <div className="empty">
          <b>No models listed.</b>Pick a server in the header and its models show up here.
        </div>
      ) : (
        <div className="labels">
          {models.map((m) => (
            <label key={m.id} className="label-row">
              <input
                type="text"
                value={aliases[m.id] ?? ''}
                placeholder={autoModelLabel(m.id)}
                onChange={(e) => set(m.id, e.target.value)}
                aria-label={`Label for ${m.id}`}
              />
              <code title={m.id}>{m.id}</code>
            </label>
          ))}
        </div>
      )}
    </section>
  )
}

function Prices({ models }: { models: Model[] }) {
  const [data, setData] = useState<PriceList | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [note, setNote] = useState<string | null>(null)

  function reload() {
    return api
      .prices()
      .then(setData)
      .catch((e: Error) => setErr(e.message))
  }
  useEffect(() => {
    reload()
  }, [])

  const mine = new Map((data?.prices ?? []).map((p) => [p.model, p]))
  // Models of the selected server first, then anything priced before that it does not list.
  const ids = [...new Set([...models.map((m) => m.id), ...mine.keys()])]

  async function save(price: ModelPrice) {
    setErr(null)
    try {
      const r = await api.setPrice(price)
      setNote(r.filled ? `${price.model}: saved, and ${r.filled} earlier result${r.filled === 1 ? '' : 's'} got a cost.` : `${price.model}: saved.`)
      await reload()
    } catch (e) {
      setErr((e as Error).message)
    }
  }
  async function clear(model: string) {
    setErr(null)
    try {
      await api.deletePrice(model)
      setNote(`${model}: price removed. Costs already recorded stay.`)
      await reload()
    } catch (e) {
      setErr((e as Error).message)
    }
  }

  return (
    <section className="settings-block">
      <h2>Prices</h2>
      <p className="lede">
        US dollars per million tokens, used to work out what each run cost. Saving applies from the next run, and fills in earlier results of that
        model that ran without a price; results that already have a cost keep it. An endpoint that bills per request itself (OpenRouter) wins over
        this table. Claude models start from Anthropic's list price{data ? ` as of ${data.builtin_as_of}` : ''}, shown greyed out.
      </p>
      {ids.length === 0 ? (
        <div className="empty">
          <b>No models listed.</b>Pick a server in the header and its models show up here.
        </div>
      ) : (
        <div className="prices">
          <div className="price-row head" aria-hidden="true">
            <span>model</span>
            <span>input $/1M</span>
            <span>output $/1M</span>
          </div>
          {ids.map((id) => (
            <PriceRow key={`${id}:${mine.get(id)?.input_per_mtok ?? ''}:${mine.get(id)?.output_per_mtok ?? ''}`} id={id} mine={mine.get(id)} builtin={data ? builtinFor(id, data.builtin) : undefined} onSave={save} onClear={clear} />
          ))}
        </div>
      )}
      {note && <div className="hint">{note}</div>}
      {err && <div className="banner error">{err}</div>}
    </section>
  )
}

function PriceRow({
  id,
  mine,
  builtin,
  onSave,
  onClear,
}: {
  id: string
  mine?: ModelPrice
  builtin?: ModelPrice
  onSave: (p: ModelPrice) => void
  onClear: (model: string) => void
}) {
  const [input, setInput] = useState(mine ? String(mine.input_per_mtok) : '')
  const [output, setOutput] = useState(mine ? String(mine.output_per_mtok) : '')
  const parse = (v: string) => (v.trim() === '' ? NaN : Number(v.replace(',', '.')))
  const i = parse(input)
  const o = parse(output)
  const valid = Number.isFinite(i) && Number.isFinite(o) && i >= 0 && o >= 0
  const dirty = !mine || i !== mine.input_per_mtok || o !== mine.output_per_mtok
  return (
    <form
      className="price-row"
      onSubmit={(e) => {
        e.preventDefault()
        if (valid && dirty) onSave({ model: id, input_per_mtok: i, output_per_mtok: o })
      }}
    >
      <code title={id}>{id}</code>
      <input
        type="text"
        inputMode="decimal"
        value={input}
        placeholder={builtin ? String(builtin.input_per_mtok) : 'no price'}
        onChange={(e) => setInput(e.target.value)}
        aria-label={`Input price for ${id}`}
      />
      <input
        type="text"
        inputMode="decimal"
        value={output}
        placeholder={builtin ? String(builtin.output_per_mtok) : 'no price'}
        onChange={(e) => setOutput(e.target.value)}
        aria-label={`Output price for ${id}`}
      />
      <button className="btn quiet" disabled={!valid || !dirty} aria-label={`Save price for ${id}`}>
        Save
      </button>
      {mine ? (
        <button type="button" className="btn quiet danger" onClick={() => onClear(id)} aria-label={`Remove price for ${id}`}>
          Remove
        </button>
      ) : (
        <span className="dim">{builtin ? 'list price' : ''}</span>
      )}
    </form>
  )
}

function GlobalSystem({ settings, onChange }: { settings: AppSettings; onChange: (s: AppSettings) => void }) {
  const [draft, setDraft] = useState(settings.global_system_prompt)
  const [err, setErr] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const dirty = draft !== settings.global_system_prompt

  async function save() {
    setBusy(true)
    setErr(null)
    try {
      onChange(await api.saveSettings({ global_system_prompt: draft }))
    } catch (e) {
      setErr((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <section className="settings-block">
      <h2>Global system prompt</h2>
      <p className="lede">
        Goes in front of every prompt's own system text, separated by a blank line, on the Bench and in the headless battery. Leave it empty to
        send each prompt as written. The full system text of every run is in its trace.
      </p>
      <textarea rows={6} value={draft} onChange={(e) => setDraft(e.target.value)} placeholder="e.g. Answer in Brazilian Portuguese. Be concise." aria-label="Global system prompt" />
      <div className="runbar">
        <button className="btn primary" onClick={save} disabled={!dirty || busy}>
          Save
        </button>
        {dirty && (
          <button className="btn quiet" onClick={() => setDraft(settings.global_system_prompt)}>
            Discard
          </button>
        )}
        <span className="hint">{settings.global_system_prompt.trim() ? `${settings.global_system_prompt.trim().length} chars active` : 'off'}</span>
      </div>
      {err && <div className="banner error">{err}</div>}
    </section>
  )
}

function Servers({ servers, onChange }: { servers: Server[]; onChange: () => void }) {
  const [form, setForm] = useState(EMPTY)
  const [err, setErr] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [editing, setEditing] = useState<Server | null>(null)

  async function add(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault()
    setBusy(true)
    setErr(null)
    try {
      if (editing) {
        await api.updateServer(editing.id, { ...form, ...(form.api_key ? { api_key: form.api_key } : {}) })
        setEditing(null)
      } else {
        await api.addServer({ ...form, api_key: form.api_key || null })
      }
      setForm(EMPTY)
      onChange()
    } catch (e) {
      setErr((e as Error).message)
    } finally {
      setBusy(false)
    }
  }
  async function remove(s: Server) {
    if (!window.confirm(`Remove server "${s.name}"? Results that reference it block the removal.`)) return
    try {
      await api.deleteServer(s.id)
      onChange()
    } catch (e) {
      setErr((e as Error).message)
    }
  }

  return (
    <section className="settings-block">
      <h2>Servers</h2>
      <p className="lede">
        Any OpenAI-compatible base URL works, local or hosted (DeepSeek, OpenRouter, OpenAI). Pick <code>anthropic</code> for Claude through the
        Anthropic API, and <code>unsloth</code> for Unsloth Studio, which exposes the loaded model and lets the bench swap it. Keys are stored
        unencrypted in the local SQLite file and sent only to the server you configure.
      </p>
      <div className="servers">
        {servers.map((s) => (
          <div key={s.id} className="server">
            <div>
              <b>{s.name}</b>
              <span className="f">
                {s.flavor}
                {s.has_key ? ', key set' : ', no key'}
              </span>
              <div className="u">{s.base_url}</div>
            </div>
            <div className="row">
              <button
                className="btn quiet"
                onClick={() => {
                  setEditing(s)
                  setForm({ name: s.name, base_url: s.base_url, api_key: '', flavor: s.flavor })
                  setErr(null)
                }}
              >
                Edit
              </button>
              <button className="btn danger" onClick={() => remove(s)}>
                Remove
              </button>
            </div>
          </div>
        ))}
        {servers.length === 0 && (
          <div className="empty">
            <b>No servers yet.</b>Add one below to start running prompts.
          </div>
        )}
        <form className="add" onSubmit={add}>
          <label className="stack">
            name
            <input type="text" required value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
          </label>
          <label className="stack">
            base URL
            <input type="text" required placeholder={PLACEHOLDER[form.flavor]} value={form.base_url} onChange={(e) => setForm({ ...form, base_url: e.target.value })} />
          </label>
          <label className="stack">
            API key {editing && '(leave blank to keep current)'}
            <input type="password" value={form.api_key} onChange={(e) => setForm({ ...form, api_key: e.target.value })} autoComplete="off" />
          </label>
          <label className="stack">
            flavor
            <select
              value={form.flavor}
              onChange={(e) => {
                const flavor = e.target.value
                // The Anthropic URL is always the same; spare the typing.
                const base_url = flavor === 'anthropic' && !form.base_url ? ANTHROPIC_URL : form.base_url
                setForm({ ...form, flavor, base_url })
              }}
            >
              <option value="openai">openai</option>
              <option value="anthropic">anthropic</option>
              <option value="unsloth">unsloth</option>
            </select>
          </label>
          <button className="btn primary" disabled={busy}>
            {editing ? 'Save' : 'Add'}
          </button>
          {editing && (
            <button type="button" className="btn quiet" onClick={() => { setEditing(null); setForm(EMPTY) }}>
              Cancel
            </button>
          )}
        </form>
        {err && <div className="banner error">{err}</div>}
      </div>
    </section>
  )
}
