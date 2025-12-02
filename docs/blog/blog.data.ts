import { createContentLoader } from 'vitepress'

interface PostSummary {
  title: string
  url: string
  date: string
  displayDate: string
  excerptHtml?: string
}

// 1) Extract first 3 non-empty, non-heading lines from the body
function oneLineExcerpt(file: { content: string; excerpt?: string }) {
  const lines = file.content.split(/\r?\n/)

  const picked: string[] = []
  for (const raw of lines) {
    const line = raw.trim()
    if (!line) continue                      // skip empty lines
    if (/^#{1,6}\s/.test(line)) continue     // skip markdown headings (#, ##, …)

    picked.push(line)
    if (picked.length === 1) break
  }

  if (picked.length === 0) {
    file.excerpt = ''
    return
  }

  // Join lines and add your fixed trailing preamble
  const body = picked.join('\n\n')
  file.excerpt = `${body}...`
}

// 2) Format dates once at build time
function formatDate(raw: string | undefined): string {
  if (!raw) return ''
  const d = new Date(raw)
  if (Number.isNaN(d.getTime())) return String(raw)
  return d.toLocaleDateString('en-SG', {
    year: 'numeric',
    month: 'short',
    day: '2-digit'
  })
}

export default createContentLoader('blog/*.md', {
  // use custom excerpt extractor
  excerpt: oneLineExcerpt,

  // sort + shrink data that goes to the client
  transform(raw): PostSummary[] {
    return raw
      .filter((page) => page.url !== '/blog/') // drop index page
      .sort((a, b) => {
        return (
          new Date(String(b.frontmatter.date)).getTime() -
          new Date(String(a.frontmatter.date)).getTime()
        )
      })
      .map((page) => ({
        title: page.frontmatter.title ?? '(untitled)',
        url: page.url,
        date: String(page.frontmatter.date ?? ''),
        displayDate: formatDate(page.frontmatter.date),
        excerptHtml: page.excerpt // this is rendered HTML for the excerpt
      }))
  }
})
