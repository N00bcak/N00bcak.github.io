import { defineConfig } from 'vitepress'
import markdownItFootnote from 'markdown-it-footnote'
import { existsSync, readdirSync, rmSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const configDir = dirname(fileURLToPath(import.meta.url))
const usedPublicFonts = new Set([
  'RecMonoCasualNerdFont-Regular.ttf',
  'RecMonoCasualNerdFont-Bold.ttf',
  'RecMonoCasualNerdFont-Italic.ttf',
  'RecMonoCasualNerdFont-BoldItalic.ttf',
])

const customElements = [
  'mjx-container',
]

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "N00bcak's Purple Cave",
  head: [
    [
      'script',
      { async: '', src: 'https://www.googletagmanager.com/gtag/js?id=VENZWYNPB8' }
    ],
    [
      'script',
      {},
      `window.dataLayer = window.dataLayer || [];
      function gtag(){dataLayer.push(arguments);}
      gtag('js', new Date());
      gtag('config', 'VENZWYNPB8');`
    ]
  ],
  description: "Blogging Things When I Feel Like It",
  appearance: true, // enable dark mode toggle
  base: '/',
  vite: {
    plugins: [{
      name: 'prune-unused-public-fonts',
      closeBundle() {
        const outputFonts = resolve(configDir, 'dist/fonts')
        if (!existsSync(outputFonts)) return

        for (const filename of readdirSync(outputFonts)) {
          if (filename.startsWith('RecMono') && !usedPublicFonts.has(filename)) {
            rmSync(resolve(outputFonts, filename))
          }
        }
      },
    }],
  },
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: 'Home', link: '/' },
      { text: 'About', link: '/about-me' },
      { text: 'CTF Writeups', link: '/writeups' },
      { text: 'Blog', link: '/blog' }
    ],
    outline: {
      level: [2, 3], // or 'deep' for all levels
      label: ''
    },
    socialLinks: [
      { 
        icon: 'github', 
        link: 'https://github.com/N00bcak' 
      },
      {
        icon: 'gmail',
        link: 'mailto:workcybeh@gmail.com'
      }
    ]
  },
  markdown: {
    theme: {
      light: 'github-light',
      dark: 'dracula',
    },
    emoji: { shortcuts: {}},
    lineNumbers: true,
    math: {
      tex: { tags: 'ams' },
    },
    config: (md) => {
      const defaultImageRenderer = md.renderer.rules.image

      md.renderer.rules.image = (tokens, idx, options, env, self) => {
        tokens[idx].attrSet('data-zoomable', '')
        return defaultImageRenderer!(tokens, idx, options, env, self)
      }

      md.use(markdownItFootnote)
    }
  },
  vue: {
    template: {
      compilerOptions: {
        isCustomElement: (tag) => customElements.includes(tag),
      },
    },
  },
})
