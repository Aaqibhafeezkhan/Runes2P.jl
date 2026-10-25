// Adds the SVG favicon, which is red in light mode and white in dark mode (it follows the
// browser's color scheme through prefers-color-scheme inside the SVG). Browsers without SVG
// favicon support (Safari) keep the red favicon.ico. documenterBaseURL is set by Documenter on
// every page, so the path is right on nested pages too.
document.addEventListener("DOMContentLoaded", function () {
  var link = document.createElement("link");
  link.rel = "icon";
  link.type = "image/svg+xml";
  link.href = documenterBaseURL + "/assets/favicon.svg";
  document.head.appendChild(link);
});
