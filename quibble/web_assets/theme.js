"use strict";
(() => {
  const key = "quibble-theme";
  const choices = new Set(["system", "light", "dark"]);
  const system = matchMedia("(prefers-color-scheme: dark)");
  let preference = "system";
  function validated(value) {
    if (value === null || choices.has(value)) return value || "system";
    console.warn("Quibble ignored an invalid saved theme preference.");
    return "system";
  }
  function storageWarning(error) {
    if (!(error instanceof DOMException) ||
        !["SecurityError", "QuotaExceededError"].includes(error.name)) throw error;
    console.warn("Quibble could not access theme preferences. Appearance still works for this page.", error);
  }
  try { preference = validated(localStorage.getItem(key)); }
  catch (error) { storageWarning(error); }
  function apply() {
    document.documentElement.dataset.theme = preference === "system" ?
      (system.matches ? "dark" : "light") : preference;
    const select = document.querySelector("[data-theme-select]");
    if (select) select.value = preference;
  }
  apply();
  system.addEventListener("change", apply);
  addEventListener("storage", event => {
    if (event.key !== key && event.key !== null) return;
    preference = validated(event.newValue);
    apply();
  });
  document.addEventListener("DOMContentLoaded", () => {
    const select = document.querySelector("[data-theme-select]");
    if (!select) return;
    select.value = preference;
    select.hidden = false;
    select.addEventListener("change", () => {
      preference = validated(select.value);
      apply();
      try { localStorage.setItem(key, preference); }
      catch (error) { storageWarning(error); }
    });
  });
})();
