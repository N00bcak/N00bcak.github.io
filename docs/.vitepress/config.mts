import { defineConfig } from 'vitepress'
import lightbox from 'vitepress-plugin-lightbox'

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
    math: true,
    theme: {
      light: 'github-light',
      dark: 'dracula',
    },
    lineNumbers: true,
    config: (md) => {
      // enable the lightbox plugin
      md.use(lightbox, {
        // optional: plugin options, can leave empty for defaults
      })
    }
  }
})
