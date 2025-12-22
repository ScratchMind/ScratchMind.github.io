---
layout: blog
title: Writing
---

# Writing

Long-form notes, derivations, and reflections from building machine learning systems from scratch.

<div class="blog-grid">
{% for post in site.blogs reversed %}
  <article class="blog-tile">
    <h2 class="blog-tile-title">
      <a href="{{ post.url }}">{{ post.title }}</a>
    </h2>

    {% if post.subtitle %}
      <p class="blog-tile-subtitle">{{ post.subtitle }}</p>
    {% endif %}

    <div class="blog-tile-meta">
      {{ post.date | date: "%B %Y" }}
    </div>
  </article>
{% endfor %}
</div>
