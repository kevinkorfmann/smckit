(function () {
  const PRETEXT_MODULE = "https://esm.sh/@chenglou/pretext@latest";

  function px(value, fallback) {
    if (!value || value === "normal") {
      return fallback;
    }
    const parsed = Number.parseFloat(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  function canvasFontFor(el) {
    const style = window.getComputedStyle(el);
    return [
      style.fontStyle || "normal",
      style.fontVariant || "normal",
      style.fontWeight || "400",
      style.fontSize || "16px",
      style.fontFamily || "system-ui",
    ].join(" ");
  }

  function preferredLineCount(textLength) {
    if (textLength < 26) return 1;
    if (textLength < 58) return 2;
    return 3;
  }

  function chooseBalancedWidth(mod, el) {
    const parent = el.parentElement;
    if (!parent) return null;
    const style = window.getComputedStyle(el);
    const fontSize = px(style.fontSize, 16);
    const lineHeight = px(style.lineHeight, fontSize * 1.3);
    const containerWidth = Math.floor(parent.getBoundingClientRect().width);

    if (!containerWidth || containerWidth < 280) return null;
    const text = (el.textContent || "").trim();
    if (!text) return null;

    const prepared = mod.prepareWithSegments(text, canvasFontFor(el));
    const targetLines = preferredLineCount(text.length);
    let best = { width: containerWidth, score: Number.POSITIVE_INFINITY };

    const minWidth = Math.max(240, Math.floor(containerWidth * 0.52));
    const step = Math.max(10, Math.floor((containerWidth - minWidth) / 18));

    for (let width = containerWidth; width >= minWidth; width -= step) {
      const stats = mod.measureLineStats(prepared, width);
      if (!stats || !stats.lineCount) continue;

      const linePenalty = Math.abs(stats.lineCount - targetLines) * 3;
      const raggedPenalty = 1 - stats.maxLineWidth / width;
      const score = linePenalty + raggedPenalty;
      if (score < best.score) {
        best = { width, score };
      }
    }

    return best.width;
  }

  function runPretext(mod) {
    const targets = document.querySelectorAll(
      ".smckit-hero h1, .rst-content h1, .rst-content h2"
    );
    targets.forEach((el) => {
      const width = chooseBalancedWidth(mod, el);
      el.style.textWrap = "balance";
      if (width !== null) {
        el.classList.add("smckit-balance-target");
        el.style.maxWidth = `${width}px`;
      }
    });
  }

  function debounce(fn, delay) {
    let timer = null;
    return function debounced() {
      window.clearTimeout(timer);
      timer = window.setTimeout(fn, delay);
    };
  }

  async function init() {
    try {
      const mod = await import(PRETEXT_MODULE);
      const refresh = debounce(() => runPretext(mod), 120);
      runPretext(mod);
      window.addEventListener("resize", refresh, { passive: true });
    } catch (error) {
      // Keep docs fully functional even if external JS fails to load.
      document.querySelectorAll(".smckit-hero h1, .rst-content h1, .rst-content h2").forEach(
        (el) => {
          el.style.textWrap = "balance";
        }
      );
      console.warn("smckit docs: pretext integration unavailable", error);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
