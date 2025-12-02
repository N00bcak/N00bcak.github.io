---
title: CTF Writeups
---

<script setup lang="ts">
import { data as posts } from './writeups.data.ts'
</script>

# CTF Writeups

I park most of my CTF notes and post-mortems here – the good runs, the disasters,
and the “why did I only see this after the CTF ended?” moments.

<div v-if="posts.length === 0" class="post-empty">
  <p>No writeups live here yet.</p>
  <p>
    Once I start migrating things over, new posts will appear automatically.
  </p>
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
