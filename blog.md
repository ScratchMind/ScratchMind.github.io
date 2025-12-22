---
layout: blog
title: Writing
---

<section class="blog-intro markdown-body">
  <h1>Writing</h1>

  <p>
    Long-form notes, derivations, and reflections from building machine learning systems from scratch.
  </p>
</section>

<section class="blog-grid">
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
</section>
