// docs/.vitepress/theme/index.ts
import DefaultTheme from 'vitepress/theme'
import './custom.css'
import Layout from './Layout.vue'

export default {
  extends: DefaultTheme,
  setup() {
    // Auto-scroll outline to active item
    if (typeof window !== 'undefined') {
      const observer = new MutationObserver(() => {
        const activeLink = document.querySelector('.VPDocAsideOutline .active')
        if (activeLink) {
          activeLink.scrollIntoView({
            behavior: 'smooth',
            block: 'nearest',
            inline: 'nearest'
          })
        }
      })

      setTimeout(() => {
        const outline = document.querySelector('.VPDocAsideOutline')
        if (outline) {
          observer.observe(outline, {
            attributes: true,
            subtree: true,
            attributeFilter: ['class']
          })
        }
      }, 1000)
    }
  },
  Layout
}