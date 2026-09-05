// jsdom intentionally does not implement media playback. Most tests do not
// care about playback internals, but app cleanup can still call pause(). Give
// the suite quiet defaults; interaction tests replace these with richer spies.
Object.defineProperty(HTMLMediaElement.prototype, "play", {
  configurable: true,
  value: () => Promise.resolve(),
  writable: true,
});

Object.defineProperty(HTMLMediaElement.prototype, "pause", {
  configurable: true,
  value: () => {},
  writable: true,
});
