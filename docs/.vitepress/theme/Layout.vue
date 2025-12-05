<script setup lang="ts">
import DefaultTheme from 'vitepress/theme'
import mediumZoom from 'medium-zoom'
import { onMounted } from 'vue'
import { useRouter } from 'vitepress'

const { Layout } = DefaultTheme
const router = useRouter()

const setupZoom = () => {
  mediumZoom('[data-zoomable]', {
    background: 'rgba(0, 0, 0, 0.9)', // plays nicely with your black theme
    margin: 24
  })
}

// run once, and again after every route change
onMounted(setupZoom)
router.onAfterRouteChange = () => setupZoom()
</script>

<template>
  <Layout>
    <template #doc-before>
      <h1 v-if="$frontmatter.title" id="title">
        {{ $frontmatter.title }}
      </h1>
    </template>
  </Layout>
</template>

<style>
.medium-zoom-overlay {
  backdrop-filter: blur(6px);
}

.medium-zoom-overlay,
.medium-zoom-image--opened {
  z-index: 999;
}
</style>
