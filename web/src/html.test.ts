import { describe, expect, it } from 'vitest'
import { extractHtml } from './html'

describe('extractHtml', () => {
  it('prefers a fenced block', () => {
    expect(extractHtml('hi\n```html\n<html><body>x</body></html>\n```\nbye')).toBe('<html><body>x</body></html>')
  })
  it('slices from doctype to closing html in prose', () => {
    expect(extractHtml('Sure: <!DOCTYPE html><html>a</html> trailing')).toBe('<!DOCTYPE html><html>a</html>')
  })
  it('falls back to the whole answer', () => {
    expect(extractHtml('  <div>partial</div> ')).toBe('<div>partial</div>')
  })
})
