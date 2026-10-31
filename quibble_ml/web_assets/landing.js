"use strict";
if (location.pathname === "/" && new URLSearchParams(location.hash.slice(1)).has("job")) {
  location.replace(`/workbench${location.hash}`);
}
