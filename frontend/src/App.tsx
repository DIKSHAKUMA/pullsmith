import { Navigate, Route, Routes } from 'react-router-dom'

import { Layout } from '@/components/Layout'
import { Dashboard } from '@/pages/Dashboard'
import { RepositoryList } from '@/pages/RepositoryList'
import { RepositoryPage } from '@/pages/RepositoryPage'
import { ReviewPage } from '@/pages/ReviewPage'
import { RunPage } from '@/pages/RunPage'

export function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="repositories" element={<RepositoryList />} />
        <Route path="repositories/:repositoryId" element={<RepositoryPage />} />
        <Route path="runs/:runId" element={<RunPage />} />
        <Route path="runs/:runId/review" element={<ReviewPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  )
}
