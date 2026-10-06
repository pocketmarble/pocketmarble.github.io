# CLAUDE.md

Personal academic website of Austin Meadows, served at https://pocketmarble.github.io.
Built on the **al-folio** Jekyll theme (README.md, INSTALL.md, CUSTOMIZE.md, FAQ.md are the upstream theme docs, not site-specific).

## Build & deploy

- Pushing to `main` triggers `.github/workflows/deploy.yml`, which builds the site and publishes it to GitHub Pages. There is no separate deploy step.
- Local preview: `docker compose up` (serves on http://localhost:8080), or `bundle exec jekyll serve` with Ruby installed.
- `_site/` is build output; never edit it.

## Where things live

- `_config.yml` — site-wide settings (name, url, nav, plugins). `_config_new.yml` is an unused upstream copy.
- `_pages/` — top-level pages. `about2.md` is the home page; `resume2.md` is the resume page (`/cv/`).
- `_projects/` — project write-ups (`/projects/<filename>/`). The micrograd, makemore*, and platoGPT pages are hand-written ML walkthroughs with in-browser demos; their data lives in `assets/json/`.
- `_bibliography/papers.bib` — publications list (rendered by jekyll-scholar on `/publications/`).
- `assets/img/`, `assets/pdf/`, `assets/json/` — static assets.

## Resume

The resume has two sources that must be kept in sync by hand:

1. **`assets/pdf/resume.pdf`** — the downloadable PDF (the icon on the resume page, via `cv_pdf: resume.pdf` in `_pages/resume2.md`). Replace it by overwriting this file; keep the filename so links don't break.
2. **`assets/json/resume.json`** — the content rendered on the page, in JSON Resume format. Loaded via `jekyll_get_json` in `_config.yml`; only the sections listed under `jsonresume:` there are shown (currently basics, work, education, publications, projects). Templates are in `_includes/resume/`.

Social/profile links: home-page icons come from `_data/socials.yml`; resume-page links come from `basics.profiles` in `resume.json`.

`_data/cv.yml` is the theme's fallback and is ignored while `resume.json` exists.

## Conventions

- Commit messages are short and informal; commit/push only when asked.
- Keep changes minimal and match the existing al-folio structure rather than adding new frameworks.
