function renderCourseDiagrams() {
  if (!window.mermaid) return;
  // Material stores its active palette on <body>, not on <html>.
  const dark = document.body?.getAttribute("data-md-color-scheme") === "slate";
  const diagrams = [...document.querySelectorAll(".mermaid")];
  for (const diagram of diagrams) {
    if (!diagram.dataset.mermaidSource) {
      diagram.dataset.mermaidSource = diagram.textContent || "";
    } else {
      // Restore source before re-rendering so theme changes do not retain old SVG colors.
      diagram.textContent = diagram.dataset.mermaidSource;
    }
    diagram.removeAttribute("data-processed");
  }

  window.mermaid.initialize({
    startOnLoad: false,
    securityLevel: "strict",
    theme: dark ? "dark" : "neutral",
    fontFamily: "inherit",
  });
  if (diagrams.length) window.mermaid.run({ nodes: diagrams });
}

if (typeof document$ !== "undefined") {
  document$.subscribe(renderCourseDiagrams);
} else {
  document.addEventListener("DOMContentLoaded", renderCourseDiagrams);
}

// Re-render diagrams when the palette toggle changes without a page navigation.
if (document.body && "MutationObserver" in window) {
  const paletteObserver = new MutationObserver((changes) => {
    if (changes.some((change) => change.attributeName === "data-md-color-scheme")) {
      renderCourseDiagrams();
    }
  });
  paletteObserver.observe(document.body, {
    attributes: true,
    attributeFilter: ["data-md-color-scheme"],
  });
}
