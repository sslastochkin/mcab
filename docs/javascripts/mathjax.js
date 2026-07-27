window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
  }
};

document$.subscribe(() => {
  MathJax.typesetPromise();
});
