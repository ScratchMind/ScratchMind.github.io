---
layout: blog
title: Writing
---

# Writing

Long-form notes, derivations, and reflections from building machine learning systems from scratch.

<ul>
{% for post in site.blogs reversed %}
  <li>
    <a href="{{ post.url }}">{{ post.title }}</a>
    <small> — {{ post.date | date: "%B %Y" }}</small>
  </li>
{% endfor %}
</ul>
