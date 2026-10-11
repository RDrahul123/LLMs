# Learn LLMs Course Website

This folder contains the deployable course-site layer. The Markdown lessons, diagrams, and example code remain in their existing module/project folders; `prepare_docs.py` assembles a generated documentation tree for MkDocs without duplicating those sources in Git.

## Run locally

From the repository root:

```bash
python -m venv .venv-course
# Windows PowerShell
.venv-course\Scripts\Activate.ps1
# macOS/Linux
# source .venv-course/bin/activate

python -m pip install -r course_site/requirements.txt
python course_site/prepare_docs.py
mkdocs serve
```

Open the local URL printed by MkDocs. The site includes full-text search, responsive module navigation, horizontally scrollable detailed diagrams and tables on phones, Mermaid diagrams, a single dark theme, and browser-local lesson progress. Progress is saved in the current browser only; no learner account or tracking service is used.

## Build for deployment

```bash
python course_site/prepare_docs.py
mkdocs build --strict
```

The static website is written to `.course-site-build/site/`. This output directory can be hosted on any static-file provider.

## GitHub Pages

The workflow in `.github/workflows/deploy-course.yml` builds and deploys the site whenever course/site files are pushed to `main`.

1. In the GitHub repository, open **Settings → Pages**.
2. Set **Build and deployment → Source** to **GitHub Actions**.
3. Push to `main` or run the workflow manually from **Actions → Deploy course site**.
4. GitHub Pages publishes at `https://rdrahul123.github.io/LLMs/`.

## Netlify

The repository includes `netlify.toml`. Create a Netlify site from this repository, use the repository root as the base directory, and deploy. The build command assembles the docs, runs a strict MkDocs build, and publishes `.course-site-build/site`.

## Vercel

The repository includes `vercel.json` with the same static build command and output directory. Import the repository into Vercel and deploy from the project root.

## Docker / Render / Railway / other containers

Build the included static-site image from the repository root:

```bash
docker build -f Dockerfile.course -t learn-llms-course .
docker run --rm -p 8080:80 learn-llms-course
```

Open `http://localhost:8080`. The image contains a built static site served by Nginx, so it can run on any platform that accepts a Dockerfile. No model API keys are needed to build or serve the learning site.

## Update lesson content

Edit the existing module/project Markdown and assets. The preparation script copies them at build time, so the website stays synchronized with the course sources. Edit `mkdocs.yml` to change navigation and `course_site/static/` for site-specific styling or scripts.
