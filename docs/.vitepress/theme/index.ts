// docs/.vitepress/theme/index.ts
import DefaultTheme from 'vitepress/theme'
import './custom.css'
import Layout from './Layout.vue'

import { nextTick, watch, onMounted, onUnmounted } from 'vue'
import { useData, useRoute } from 'vitepress'
import mermaidTinyUrl from '@mermaid-js/tiny/dist/mermaid.tiny.js?url'

type Mermaid = {
  initialize: (config: Record<string, unknown>) => void
  render: (id: string, source: string) => Promise<{ svg: string }>
}

declare global {
  interface Window {
    mermaid?: Mermaid
  }
}

let mermaidLoader: Promise<Mermaid> | undefined
let diagramId = 0

const loadMermaid = () => {
  if (window.mermaid) return Promise.resolve(window.mermaid)
  if (mermaidLoader) return mermaidLoader

  mermaidLoader = new Promise((resolve, reject) => {
    const script = document.createElement('script')
    script.src = mermaidTinyUrl
    script.onload = () => window.mermaid
      ? resolve(window.mermaid)
      : reject(new Error('Mermaid loaded without exposing its API'))
    script.onerror = () => reject(new Error('Failed to load Mermaid'))
    document.head.appendChild(script)
  })

  return mermaidLoader
}

export default {
  extends: DefaultTheme,

  setup() {
    const { isDark } = useData()
    const route = useRoute()

    let observer: MutationObserver | undefined

    const renderMermaid = async () => {
      await nextTick()
      const blocks = document.querySelectorAll<HTMLElement>('.language-mermaid')
      if (!blocks.length) return

      const mermaid = await loadMermaid()
      mermaid.initialize({
        startOnLoad: false,
        securityLevel: 'strict',
        theme: isDark.value ? 'dark' : 'default',
      })

      for (const block of blocks) {
        const source = block.dataset.mermaidSource
          ?? block.querySelector('code')?.textContent
        if (!source) continue

        block.dataset.mermaidSource = source

        try {
          const { svg } = await mermaid.render(`mermaid-${diagramId++}`, source)
          block.innerHTML = svg
          block.classList.add('mermaid-rendered')
        } catch (error) {
          console.error('Failed to render Mermaid diagram', error)
        }
      }
    }

    onMounted(() => {
      // Initial Mermaid render
      renderMermaid()

      // Auto-scroll outline to active item
      observer = new MutationObserver(() => {
        const activeLink = document.querySelector('.VPDocAsideOutline .active')

        if (activeLink) {
          activeLink.scrollIntoView({
            behavior: 'smooth',
            block: 'nearest',
            inline: 'nearest',
          })
        }
      })

      setTimeout(() => {
        const outline = document.querySelector('.VPDocAsideOutline')

        if (outline && observer) {
          observer.observe(outline, {
            attributes: true,
            subtree: true,
            attributeFilter: ['class'],
          })
        }
      }, 1000)
    })

    // Re-render Mermaid when VitePress switches light/dark mode
    watch(
      () => isDark.value,
      () => {
        renderMermaid()
      },
    )

    // Re-render Mermaid after page navigation
    watch(
      () => route.path,
      () => {
        renderMermaid()
      },
    )

    onUnmounted(() => {
      observer?.disconnect()
    })
  },

  Layout,
}
