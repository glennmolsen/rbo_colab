# Go Unit Test Isolation

- Unit tests must not depend on host cert files, external DNS, or network availability.
- Use per-test local mock transport/state; avoid shared global responder registries and global reset patterns.
- Avoid process-global coupling in parallelizable tests (for example, broad env mutation patterns); prefer pure helper testing where possible.
- Add `t.Parallel()` when setup is isolated and deterministic.
- Keep test names behavior-first and assertions focused on observable outcomes.
- In handoff notes, summarize unit-test coverage for changed behavior: covered paths, uncovered paths, and rationale for any deferred coverage.
- Follow AGENTS seam precedence: prefer public black-box seams first; internal test-only seams require explicit rationale.
- For package public behavior (for example, `ipam`), prefer external-package `*_test` coverage before adding same-package seam hooks.
