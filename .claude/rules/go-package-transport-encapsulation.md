# Go Package Transport Encapsulation

- Applies to handwritten Go code only; excludes generated and mock paths/files.
- Public constructors for service clients should provide production-safe transport defaults.
- Standard usage should not require callers to wire TLS/mTLS/cert-path details directly.
- Allow explicit HTTP client or transport injection as an override seam, primarily for tests.
- Keep transport policy centralized in package internals; avoid spreading transport branching across business logic.
- When testing transport behavior, prefer public constructor/injection seams from external tests before adding package-internal test-only seams.
- Preserve existing public API signatures unless a change is explicitly requested.
