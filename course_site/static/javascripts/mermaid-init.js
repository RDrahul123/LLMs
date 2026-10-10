function renderCourseDiagrams() {
  if (!window.mermaid) return;
  const diagrams = [...document.querySelectorAll(".mermaid")];
  for (const diagram of diagrams) {
    if (!diagram.dataset.mermaidSource) {
      diagram.dataset.mermaidSource = diagram.textContent || "";
    } else {
      // Restore source before client-side navigation re-renders to avoid nesting old SVGs.
      diagram.textContent = diagram.dataset.mermaidSource;
    }
    diagram.removeAttribute("data-processed");
  }

  window.mermaid.initialize({
    startOnLoad: false,
    securityLevel: "strict",
    theme: "dark",
    fontFamily: "inherit",
  });
  if (diagrams.length) window.mermaid.run({ nodes: diagrams });
}

if (typeof document$ !== "undefined") {
  document$.subscribe(renderCourseDiagrams);
} else {
  document.addEventListener("DOMContentLoaded", renderCourseDiagrams);
}
