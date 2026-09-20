/**
 * A minimal unified-diff renderer.
 *
 * Written rather than pulled from a package: the input is already a unified diff, so rendering
 * it is line classification and nothing more. A diff library would add a dependency and a
 * bundle cost for behaviour we do not need.
 *
 * Long diffs are collapsed by default. A reviewer scrolling a thousand lines stops reading, so
 * the size is stated and expansion is explicit.
 */

import { useState } from 'react'

const COLLAPSE_ABOVE = 120

type LineKind = 'added' | 'removed' | 'header' | 'hunk' | 'context'

function classify(line: string): LineKind {
  if (line.startsWith('+++') || line.startsWith('---')) return 'header'
  if (line.startsWith('@@')) return 'hunk'
  if (line.startsWith('+')) return 'added'
  if (line.startsWith('-')) return 'removed'
  return 'context'
}

const STYLES: Record<LineKind, string> = {
  added: 'bg-emerald-500/10 text-emerald-300',
  removed: 'bg-rose-500/10 text-rose-300',
  header: 'text-neutral-500',
  hunk: 'bg-sky-500/10 text-sky-300',
  context: 'text-neutral-400',
}

export function DiffViewer({ diffText }: { diffText: string }) {
  const [expanded, setExpanded] = useState(false)

  if (!diffText.trim()) {
    return <p className="text-sm text-neutral-500">No diff available.</p>
  }

  const lines = diffText.split('\n')
  const collapsed = lines.length > COLLAPSE_ABOVE && !expanded
  const visible = collapsed ? lines.slice(0, COLLAPSE_ABOVE) : lines

  return (
    <div className="overflow-hidden rounded border border-neutral-800">
      <pre className="mono max-h-[32rem] overflow-auto text-xs leading-relaxed">
        {visible.map((line, index) => (
          <div
            // Index is a safe key here: the diff is static for the lifetime of this view.
            key={`${index}-${line.slice(0, 12)}`}
            className={`px-3 ${STYLES[classify(line)]}`}
          >
            {line || ' '}
          </div>
        ))}
      </pre>

      {lines.length > COLLAPSE_ABOVE && (
        <button
          type="button"
          onClick={() => setExpanded(!expanded)}
          className="mono w-full border-t border-neutral-800 bg-neutral-900/60 px-3 py-2 text-left text-xs text-sky-300 hover:bg-neutral-900"
        >
          {collapsed
            ? `show all ${lines.length} lines`
            : `collapse to ${COLLAPSE_ABOVE} lines`}
        </button>
      )}
    </div>
  )
}
