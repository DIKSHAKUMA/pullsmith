import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { DiffViewer } from '@/components/DiffViewer'
import { RiskBadge } from '@/components/RiskBadge'

const DIFF = [
  '--- a/app/profile.py',
  '+++ b/app/profile.py',
  '@@ -1,4 +1,4 @@',
  ' class ProfileService:',
  "-        raise ValueError('email required')",
  "+        raise ValidationError('email required')",
].join('\n')

describe('DiffViewer', () => {
  it('renders added and removed lines', () => {
    render(<DiffViewer diffText={DIFF} />)

    expect(screen.getByText(/\+.*ValidationError/)).toBeDefined()
    expect(screen.getByText(/-.*ValueError/)).toBeDefined()
  })

  it('handles an empty diff without crashing', () => {
    render(<DiffViewer diffText="" />)

    expect(screen.getByText('No diff available.')).toBeDefined()
  })

  it('collapses a long diff and states the real size', () => {
    const long = Array.from({ length: 300 }, (_, index) => `+line ${index}`).join('\n')

    render(<DiffViewer diffText={long} />)

    // A reviewer scrolling a thousand lines stops reading, so expansion is explicit.
    expect(screen.getByText('show all 300 lines')).toBeDefined()
  })
})

describe('RiskBadge', () => {
  it('shows the level and score', () => {
    render(<RiskBadge level="HIGH" score={12} />)

    expect(screen.getByText(/high risk/)).toBeDefined()
    expect(screen.getByText('(12)')).toBeDefined()
  })

  it('describes itself as a heuristic, not a guarantee', () => {
    render(<RiskBadge level="LOW" score={1} />)

    const badge = screen.getByTitle(/not a security or correctness guarantee/i)

    expect(badge).toBeDefined()
  })
})
