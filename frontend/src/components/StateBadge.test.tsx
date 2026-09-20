import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { StateBadge, toneFor } from '@/components/StateBadge'

describe('toneFor', () => {
  it('marks human gates distinctly from running states', () => {
    expect(toneFor('WAITING_FOR_APPROVAL')).toBe('gate')
    expect(toneFor('WAITING_FOR_PLAN_REVIEW')).toBe('gate')
    expect(toneFor('TESTING')).toBe('running')
  })

  it('separates success from failure', () => {
    expect(toneFor('COMPLETED')).toBe('done')
    expect(toneFor('PR_CREATED')).toBe('done')
    expect(toneFor('FAILED')).toBe('bad')
    expect(toneFor('CANCELLED')).toBe('bad')
  })
})

describe('StateBadge', () => {
  it('renders a readable label instead of the raw enum', () => {
    render(<StateBadge state="WAITING_FOR_APPROVAL" />)
    expect(screen.getByText('waiting for approval')).toBeDefined()
  })
})
