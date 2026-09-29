/** Mirror of core/html_extractor.extract_html for the live iframe. */
export function extractHtml(answer: string): string {
  const m = /```[a-zA-Z0-9_-]*\n([\s\S]*?)```/.exec(answer)
  const code = (m ? m[1] : answer).trim()
  const low = code.toLowerCase()
  let start = low.indexOf('<!doctype')
  if (start < 0) start = low.indexOf('<html')
  const end = low.lastIndexOf('</html>')
  if (start >= 0 && end > start) return code.slice(start, end + '</html>'.length)
  return code
}
