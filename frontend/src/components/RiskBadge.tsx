/**
 * Risk level badge.
 *
 * The tooltip says what the score is *not*. A number next to the word "risk" invites a reader
 * to treat it as a safety verdict, and it is a heuristic for directing attention.
 */

const TONE: Record<string, string> = {
  LOW: 'bg-emerald-500/10 text-emerald-300 ring-emerald-500/30',
  MEDIUM: 'bg-amber-500/10 text-amber-300 ring-amber-500/30',
  HIGH: 'bg-rose-500/10 text-rose-300 ring-rose-500/30',
}

export function RiskBadge({ level, score }: { level: string; score: number }) {
  const tone = TONE[level] ?? TONE.MEDIUM

  return (
    <span
      title="A heuristic to direct reviewer attention. Not a security or correctness guarantee."
      className={`mono inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ring-1 ring-inset ${tone}`}
    >
      {level.toLowerCase()} risk
      <span className="opacity-60">({score})</span>
    </span>
  )
}
