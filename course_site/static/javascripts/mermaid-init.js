function renderCourseDiagrams() {
  if (!window.mermaid) return;
  const dark = document.documentElement.getAttribute("data-md-color-scheme") === "slate";
  window.mermaid.initialize({
    startOnLoad: false,
    securityLevel: "strict",
    theme: dark ? "dark" : "neutral",
    fontFamily: "inherit",
  });
  window.mermaid.run({ querySelector: ".mermaid" });
}

if (typeof document$ !== "undefined") {
  document$.subscribe(renderCourseDiagrams);
} else {
  document.addEventListener("DOMContentLoaded", renderCourseDiagrams);
}
