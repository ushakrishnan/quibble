"use strict";
(() => {
  const preferences = {
    theme: {fallback: "system", choices: ["system", "light", "dark"]},
    accent: {fallback: "blue", choices: ["blue", "teal", "indigo", "violet", "fuchsia", "slate"]},
    contrast: {fallback: "standard", choices: ["standard", "high"]},
    rail: {fallback: "expanded", choices: ["expanded", "collapsed"]}
  };
  const system = matchMedia("(prefers-color-scheme: dark)");
  const values = {};
  let warning = "";
  function warn(message, error) {
    warning = message;
    console.warn(message, ...(error ? [error] : []));
    document.querySelectorAll("[data-appearance-warning]").forEach(node => {
      node.textContent = warning;
      node.hidden = false;
    });
  }
  function validated(name, value) {
    const preference = preferences[name];
    if (value === null) return preference.fallback;
    if (preference.choices.includes(value)) return value;
    warn(`Quibble ignored an invalid saved ${name} preference; using ${preference.fallback}.`);
    return preference.fallback;
  }
  function storageWarning(error) {
    if (!(error instanceof DOMException) ||
        !["SecurityError", "QuotaExceededError"].includes(error.name)) throw error;
    warn("Quibble could not access appearance preferences. Changes still work for this page but may not be saved.", error);
  }
  for (const name of Object.keys(preferences)) {
    values[name] = preferences[name].fallback;
    try { values[name] = validated(name, localStorage.getItem(`quibble-${name}`)); }
    catch (error) { storageWarning(error); }
  }
  function apply() {
    for (const name of Object.keys(preferences)) {
      document.documentElement.dataset[name] = name === "theme" && values.theme === "system" ?
        (system.matches ? "dark" : "light") : values[name];
      document.querySelectorAll(`[data-${name}-select]`).forEach(select => {
        select.value = values[name];
        select.hidden = false;
      });
    }
    document.querySelectorAll("[data-rail-toggle]").forEach(button => {
      button.setAttribute("aria-pressed", String(values.rail === "collapsed"));
    });
  }
  apply();
  system.addEventListener("change", apply);
  addEventListener("storage", event => {
    if (event.key !== null && !Object.keys(preferences).some(name => event.key === `quibble-${name}`)) return;
    if (event.storageArea) {
      try { if (event.storageArea !== localStorage) return; }
      catch (error) { storageWarning(error); }
    }
    for (const name of Object.keys(preferences)) {
      if (event.key === null || event.key === `quibble-${name}`)
        values[name] = validated(name, event.key === null ? null : event.newValue);
    }
    apply();
    document.dispatchEvent(new Event("quibble-appearance"));
  });
  document.addEventListener("DOMContentLoaded", () => {
    function save(name, value) {
      values[name] = validated(name, value);
      apply();
      try { localStorage.setItem(`quibble-${name}`, values[name]); }
      catch (error) { storageWarning(error); }
      document.dispatchEvent(new Event("quibble-appearance"));
    }
    apply();
    if (warning) document.querySelectorAll("[data-appearance-warning]").forEach(node => {
      node.textContent = warning;
      node.hidden = false;
    });
    for (const name of Object.keys(preferences)) {
      document.querySelectorAll(`[data-${name}-select]`).forEach(select =>
        select.addEventListener("change", () => save(name, select.value)));
    }
    document.querySelectorAll("[data-rail-toggle]").forEach(button => {
      button.addEventListener("click", () =>
        save("rail", values.rail === "collapsed" ? "expanded" : "collapsed"));
    });
  });
})();
