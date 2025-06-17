## Condor unit test structure
- `standard/` - "standard" software-engineering unit tests, ie. tests for constructors, deterministic logic, etc. (Should mostly be fast; shouldn't involve testing details of model behavior, etc.)
- `model/`
  - `unit_tests/` - tests for smaller pieces of the model
  - `integration_tests/` - may require compiling GFI methods for large genjax programs
- `inference/`
  - `unit_tests/` - tests for smaller pieces of inference (e.g. individual gibbs moves)
  - `integration_tests/` - these may be pretty slow due to compile times, etc.
