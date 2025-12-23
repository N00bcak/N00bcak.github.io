---
title: Blog
---

<script setup lang="ts">
import { data as posts } from './blog.data.ts'
</script>

Welcome to my blog! Here I share my thoughts on various topics including technology, programming, and personal experiences.

<div v-if="posts.length === 0" class="post-empty">
  <p>No blogposts live here yet. Stay tuned!</p>
</div>

<div v-else class="post-list">
  <article
    v-for="post in posts"
    :key="post.url"
    class="post-item"
  >
    <p v-if="post.displayDate" class="post-date">
      {{ post.displayDate }}
    </p>
    <h2 class="post-title">
      <a :href="post.url">
        {{ post.title }}
      </a>
    </h2>
    <div
      v-if="post.excerptHtml"
      class="post-excerpt"
      v-html="post.excerptHtml"
    />
  </article>
</div>
