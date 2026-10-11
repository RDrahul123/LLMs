function makeCourseDiagramsScrollable() {
  for (const image of document.querySelectorAll('.md-typeset img[src*="assets/diagrams/"]')) {
    if (image.closest(".diagram-scroll")) continue;

    const wrapper = document.createElement("div");
    wrapper.className = "diagram-scroll";
    wrapper.setAttribute("role", "region");
    wrapper.setAttribute("aria-label", "Scrollable course diagram");
    wrapper.tabIndex = 0;

    const paragraph = image.parentElement?.tagName === "P" ? image.parentElement : null;
    if (paragraph && paragraph.childElementCount === 1) {
      paragraph.replaceWith(wrapper);
      wrapper.append(paragraph);
    } else {
      image.before(wrapper);
      wrapper.append(image);
    }
  }
}

function renderCourseDiagrams() {
  makeCourseDiagramsScrollable();
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
