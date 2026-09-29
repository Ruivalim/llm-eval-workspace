/// <reference types="vitest/config" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: true, // escuta em 0.0.0.0 pra abrir de outra máquina
    port: 5173,
    proxy: { '/api': { target: 'http://127.0.0.1:8600', changeOrigin: false } },
  },
  build: { outDir: 'dist', emptyOutDir: true },
  test: { environment: 'jsdom', globals: false, include: ['src/**/*.test.ts', 'src/**/*.test.tsx'] },
})
