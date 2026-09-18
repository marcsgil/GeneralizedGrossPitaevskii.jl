# Changelog

## v0.2.1

### Fixes

- Restore backend-specific default random-number generation for stochastic simulations.
  In v0.2.0, the solver passed Julia's CPU RNG explicitly by default, causing GPU noise
  arrays to be generated on the CPU and copied to the GPU at every noise update.
  The default is now `rng=nothing`, which delegates to `randn!(array)` so the array
  backend chooses its generator. Explicit RNGs remain supported.

### Compatibility

- No breaking API changes. Existing calls that omit `rng` automatically benefit from
  the fix. Calls that supply an explicit RNG retain that choice; use an RNG compatible
  with the noise arrays. Default GPU random sequences may differ from v0.2.0.
