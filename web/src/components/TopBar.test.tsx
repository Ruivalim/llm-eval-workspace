import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import type { Prefs } from '../store'
import { TopBar } from './TopBar'

afterEach(cleanup)

const prefs: Prefs = {
  serverId: 1,
  model: 'model-a',
  locale: 'en',
  promptId: null,
  temperature: 0.7,
  topP: 1,
  effort: 'default',
  timeoutS: 60,
  parallel: 1,
  aliases: {},
  view: 'bench',
}

it('reloads models from model selector', () => {
  const refreshModels = vi.fn()
  render(
    <TopBar
      servers={[{ id: 1, name: 'local', flavor: 'openai', base_url: 'http://localhost', has_key: false }]}
      prefs={prefs}
      update={() => {}}
      locales={['en']}
      models={[{ id: 'model-a', loaded: true }]}
      modelsError={null}
      loadingModels={false}
      refreshModels={refreshModels}
      running={false}
      onStop={() => {}}
    />,
  )

  fireEvent.click(screen.getByRole('button', { name: 'Reload models' }))
  expect(refreshModels).toHaveBeenCalledOnce()
})
