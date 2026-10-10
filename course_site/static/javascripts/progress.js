(() => {
  const STORAGE_KEY = "learn-llms-completed-pages-v1";

  function loadCompleted() {
    try {
      const value = JSON.parse(window.localStorage.getItem(STORAGE_KEY) || "[]");
      return new Set(Array.isArray(value) ? value : []);
    } catch {
      return new Set();
    }
  }

  function renderProgress() {
    const content = document.querySelector(".md-content__inner");
    if (!content) return;

    const pageLinks = [...document.querySelectorAll(".md-nav__link[href]")]
      .map((link) => {
        try {
          const url = new URL(link.href, window.location.href);
          return url.origin === window.location.origin ? url.pathname : null;
        } catch {
          return null;
        }
      })
      .filter(Boolean);
    const lessons = [...new Set(pageLinks)];
    const current = window.location.pathname;
    if (!lessons.includes(current)) lessons.push(current);

    const completed = loadCompleted();
    const completedCount = lessons.filter((page) => completed.has(page)).length;
    const percent = lessons.length ? Math.round((completedCount / lessons.length) * 100) : 0;
    const alreadyRendered = content.querySelector(":scope > .course-progress");
    if (alreadyRendered) alreadyRendered.remove();

    const panel = document.createElement("section");
    panel.className = "course-progress";
    panel.setAttribute("aria-label", "Course reading progress");

    const copy = document.createElement("div");
    copy.className = "course-progress__copy";
    const title = document.createElement("strong");
    title.textContent = "Your course progress";
    const count = document.createElement("span");
    count.className = "course-progress__count";
    count.textContent = `${completedCount} of ${lessons.length} lessons · ${percent}%`;
    copy.append(title, count);

    const actions = document.createElement("div");
    actions.className = "course-progress__actions";
    const mark = document.createElement("button");
    const isComplete = completed.has(current);
    mark.type = "button";
    mark.setAttribute("aria-pressed", String(isComplete));
    mark.textContent = isComplete ? "✓ Lesson complete" : "Mark lesson complete";
    mark.addEventListener("click", () => {
      const next = loadCompleted();
      if (next.has(current)) next.delete(current);
      else next.add(current);
      try {
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify([...next]));
      } catch {
        count.textContent = "Progress could not be saved in this browser";
      }
      renderProgress();
    });

    const reset = document.createElement("button");
    reset.type = "button";
    reset.textContent = "Reset";
    reset.setAttribute("aria-label", "Reset saved course progress");
    reset.addEventListener("click", () => {
      try {
        window.localStorage.removeItem(STORAGE_KEY);
      } catch {
        // Keep the control usable if browser storage is disabled.
      }
      renderProgress();
    });
    actions.append(mark, reset);

    const progress = document.createElement("progress");
    progress.max = lessons.length || 1;
    progress.value = completedCount;
    progress.setAttribute("aria-label", `${percent}% of course pages marked complete`);

    panel.append(copy, actions, progress);
    content.prepend(panel);
  }

  if (typeof document$ !== "undefined") {
    document$.subscribe(renderProgress);
  } else {
    document.addEventListener("DOMContentLoaded", renderProgress);
  }
})();
