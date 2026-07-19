---
layout: page
title: projects
permalink: /projects/
description: A selection of technical projects spanning machine learning and computational research.
nav: true
nav_order: 3
display_categories: [nanoGPT, makemore, micrograd]
horizontal: false
---

<!-- pages/projects.md -->
<div class="projects">
{% if site.enable_project_categories and page.display_categories %}
  <!-- Display categorized projects -->
  {% for category in page.display_categories %}
  <a id="{{ category }}" href=".#{{ category }}">
    <h2 class="category">{{ category }}</h2>
  </a>
  {% assign categorized_projects = site.projects | where: "category", category %}
  {% assign sorted_projects = categorized_projects | sort: "importance" %}
  <!-- Generate cards for each project -->
  {% if page.horizontal %}
  <div class="container">
    <div class="row row-cols-1 row-cols-md-2">
    {% for project in sorted_projects %}
      {% include projects_horizontal.liquid %}
    {% endfor %}
    </div>
  </div>
  {% else %}
  <div class="row row-cols-2 row-cols-md-3 row-cols-lg-4">
    {% for project in sorted_projects %}
      {% include projects.liquid %}
    {% endfor %}
  </div>
  {% endif %}
  {% endfor %}

{% else %}

<!-- Display projects without categories -->

{% assign sorted_projects = site.projects | sort: "importance" %}

  <!-- Generate cards for each project -->

{% if page.horizontal %}

  <div class="container">
    <div class="row row-cols-1 row-cols-md-2">
    {% for project in sorted_projects %}
      {% include projects_horizontal.liquid %}
    {% endfor %}
    </div>
  </div>
  {% else %}
  <div class="row row-cols-1 row-cols-md-3">
    {% for project in sorted_projects %}
      {% include projects.liquid %}
    {% endfor %}
  </div>
  {% endif %}
{% endif %}
</div>

<style>
/* compact project tiles — the whole set should fit above the fold */
.projects .col { padding-bottom: 0.75rem; }

.projects .card {
  border-radius: 6px;
  overflow: hidden;
}

/* the thumbnails have wildly different aspect ratios (1:1 through 5:1);
   pin them to one banner height and centre-crop so the grid stays even */
.projects .card img,
.projects .card .card-img-top {
  height: 96px;
  width: 100%;
  object-fit: cover;
  object-position: center;
  display: block;
}

.projects .card-body { padding: 0.6rem 0.7rem 0.7rem; }

.projects .card-title {
  font-size: 0.95rem;
  font-weight: 600;
  line-height: 1.25;
  margin-bottom: 0.25rem;
  /* clamp so a long title can't stretch one tile taller than its row */
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.projects .card-text {
  font-size: 0.76rem;
  line-height: 1.35;
  margin-bottom: 0;
  color: var(--global-text-color-light);
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

/* the category rule was eating ~90px of vertical space per section */
.projects h2.category {
  font-size: 1.05rem;
  padding-top: 0;
  margin-top: 1.1rem;
  margin-bottom: 0.6rem;
}
.projects h2.category:first-of-type { margin-top: 0.4rem; }

/* the empty half of the row after the last card is dead space, not a card */
.projects .row { margin-bottom: 0; }
</style>
