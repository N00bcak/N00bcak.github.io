import { defineConfig } from 'vitepress'
import lightbox from 'vitepress-plugin-lightbox'
import mathjax3 from 'markdown-it-mathjax3'

const customElements = [
  'mjx-container',
]

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "N00bcak's Purple Cave",
  description: "Blogging Things When I Feel Like It",
  appearance: true, // enable dark mode toggle
  base: '/',
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: 'Home', link: '/' },
      { text: 'About', link: '/about-me' },
      { text: 'CTF Writeups', link: '/writeups' },
      { text: 'Blog', link: '/blog' }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/N00bcak' }
    ]
  },
  markdown: {
    theme: {
      light: 'github-light',
      dark: 'dracula',
    },
    emoji: { shortcuts: {}},
    lineNumbers: true,
    config: (md) => {
      // enable the lightbox plugin
      md.use(lightbox, {
        // optional: plugin options, can leave empty for defaults
      });
      md.use(mathjax3, {
        tex: { tags: 'ams' } // or 'all'
      });
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
