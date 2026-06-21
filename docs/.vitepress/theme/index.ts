// docs/.vitepress/theme/index.ts
import DefaultTheme from 'vitepress/theme'
import './custom.css'
import Layout from './Layout.vue'

import { nextTick, watch, onMounted, onUnmounted } from 'vue'
import { useData, useRoute } from 'vitepress'

import { createMermaidRenderer } from 'vitepress-mermaid-renderer'
import 'vitepress-mermaid-renderer/css'

export default {
  extends: DefaultTheme,

  setup() {
    const { isDark } = useData()
    const route = useRoute()

    let observer: MutationObserver | undefined

    const renderMermaid = async () => {
      await nextTick()

      createMermaidRenderer({
        theme: isDark.value ? 'dark' : 'default',
      })
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