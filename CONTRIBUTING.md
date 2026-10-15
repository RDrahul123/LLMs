# Contributing to Learn LLMs

Thank you for helping improve this course. Contributions that make LLM concepts clearer, examples more reliable, and the learning experience more accessible are welcome.

## Ways to contribute

- Report a reproducible bug, broken link, or incorrect explanation.
- Improve a lesson, diagram, exercise, project, or glossary entry.
- Fix an example or add tests for an existing project.
- Suggest a new topic or learning activity.
- Improve accessibility, navigation, or the course website.

For a substantial change, open an issue first so we can discuss its scope and approach. For a small typo or clearly scoped fix, a pull request is welcome directly.

## Before you start

1. Search [existing issues](https://github.com/RDrahul123/LLMs/issues) and pull requests to avoid duplicating work.
2. Read the relevant module, project README, and nearby examples to understand the intended level and style.
3. For a new feature or broad content change, open an issue describing the problem, proposed scope, and who it helps.
4. Do not include secrets, private data, copyrighted material you do not have permission to contribute, or content copied from other sources without attribution and compatible licensing.

## Set up your workspace

Fork the repository on GitHub, then clone your fork:

```bash
git clone https://github.com/<your-github-username>/LLMs.git
cd LLMs
git remote add upstream https://github.com/RDrahul123/LLMs.git
git switch -c docs/describe-your-change
```

Use a descriptive branch name such as `docs/improve-rag-lesson`, `fix/project-install`, or `feat/evaluation-example`. Keep changes focused so they are easier to review.

Python examples may require only their module's or project's `requirements.txt`. Avoid installing the root `requirements.txt` unless you need it; it includes optional ML and GPU-related packages.

To build the course website locally, install its smaller set of dependencies:

```bash
python -m pip install -r course_site/requirements.txt
python course_site/prepare_docs.py
python -m mkdocs serve
```

Open the local address printed by MkDocs. The generated documentation and site are placed under `.course-site-build/`; edit the source Markdown and assets instead of generated copies.

## Make a change

### Course content

- Put explanations in the appropriate module or project rather than creating a second source of truth.
- Use clear headings, concise paragraphs, tables, and small, runnable examples where they improve understanding.
- Explain assumptions, limitations, and likely failure modes; do not present model output as guaranteed correct.
- Keep commands and APIs current where possible, and label provider-specific or hardware-dependent instructions.
- Link related lessons using relative paths so links work both in the repository and on the course website.
- If you add or rename a lesson, update `mkdocs.yml` navigation and any relevant links from the root README, learning paths, or neighboring lessons.
- Update `GLOSSARY.md` when adding an important term used across the course.

### Python and project code

- Follow the surrounding code's conventions and keep examples approachable for the module's intended audience.
- Prefer explicit inputs, useful errors, and small focused functions over hidden side effects.
- Read credentials from environment variables or an untracked local `.env`; never commit real keys, tokens, or personal data. Update `.env.example` with placeholder variable names only when needed.
- Keep dependencies scoped to the relevant module or project where practical. If a new dependency is needed, add it to the appropriate `requirements.txt` and explain its use.
- Do not make tests depend on a paid API, network access, or a real secret unless the test is explicitly marked as an optional integration test.
- Avoid commands that can make destructive, external, or billable changes without clear safeguards and user confirmation.

### Diagrams and media

- Keep diagrams readable at the size they appear in the lesson and provide descriptive alternative text for images.
- Prefer editable source formats already used in the relevant area of the repository.
- Ensure contributed media is original, appropriately licensed, or otherwise permitted for redistribution; include attribution where required.

## Validate your changes

From the repository root, run the documentation checks:

```bash
python scripts/check_course_docs.py
```

This checks local Markdown links, Python syntax, and SVG XML without importing optional ML dependencies. If you changed course pages, navigation, site preparation, or styling, also build the site strictly:

```bash
python -m pip install -r course_site/requirements.txt
python course_site/prepare_docs.py
python -m mkdocs build --strict
```

If you changed Python behavior, run the relevant tests as well. For example, the ContextIQ project has tests under `projects/03-document-qa/tests/`; install that project's requirements before running them:

```bash
python -m pip install -r projects/03-document-qa/requirements.txt
python -m pytest projects/03-document-qa/tests/
```

Follow the module or project README for other test and run commands. If a check cannot be run, say why in your pull request rather than implying it passed.

## Submit a pull request

1. Review your changes and ensure generated files, local environment files, and credentials are not included.
2. Run the relevant checks above.
3. Commit the focused change with a clear message, for example `docs: clarify RAG retrieval flow`.
4. Push your branch to your fork and open a pull request against `RDrahul123/LLMs:main`.
5. Complete the pull request description with the problem addressed, the change made, related issues, and checks run. Include screenshots for meaningful visual changes.
6. Respond to review feedback and update the pull request until it is ready to merge.

Please do not include unrelated formatting changes or generated site output. Maintainers may ask for revisions, adjust scope, or close a contribution that does not fit the course.

## Reporting issues

When opening a bug report, include:

- What you expected and what happened.
- Steps to reproduce, with a minimal example where possible.
- The affected file, lesson, or project.
- Relevant OS, Python version, and dependency versions.
- Error output with API keys, personal information, and other secrets removed.

For an idea or content request, explain the learner problem it solves and the intended audience. Use the repository's issue templates when available.

If you believe you have found a security vulnerability, do not post exploit details or secrets in a public issue. Use GitHub's private vulnerability reporting for this repository if it is enabled, or contact the maintainer privately through GitHub.

## Community expectations

Be respectful, constructive, and open to different backgrounds and experience levels. Critique ideas and code, not people. Help make the course welcoming and useful to learners.

## License

By submitting a contribution, you agree that it may be distributed under this repository's [MIT License](LICENSE). Ensure you have the right to contribute the material.
