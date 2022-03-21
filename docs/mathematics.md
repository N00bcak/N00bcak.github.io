---
layout: page
title: Mathematics
permalink: /math/
---

Most of my work involves organising the content that I have learnt in ways that make the most sense to me.

Such endeavors include:
<ul>
    {% for doc in site.texdocs %}
        {% if doc.pdf_url %}
            <li><a href="{{doc.pdf_url}}">{{doc.title | markdownify}}</a></li>
        {% endif %}
    {% endfor %}
</ul>

Giving credit to the things that helped me learn and compile this document:
- Hoffman and Kunze's Linear Algebra
- My last brain cell
