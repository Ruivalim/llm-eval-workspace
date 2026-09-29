import type { ModelPrice } from './api'

/** Built-in list price for a Claude id, exact or with a dated snapshot suffix.
 * Never a looser prefix: claude-opus-5 must not price claude-opus-5-5. Mirrors
 * llm_eval/core/pricing.py `builtin_price`. */
export function builtinFor(id: string, builtin: ModelPrice[]): ModelPrice | undefined {
  const bare = id.replace(/-\d{8}$/, '')
  return builtin.find((b) => b.model === id) ?? builtin.find((b) => b.model === bare)
}
