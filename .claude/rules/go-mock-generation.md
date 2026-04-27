# SDK Client Tests — use httpmock, not real HTTP calls

Unit tests for any code that wraps an SDK or HTTP client must use `github.com/jarcoal/httpmock`
with `NewMockTransport()` to intercept calls. Do not write tests that make real outbound requests
or assert on auth/network errors as a proxy for coverage.

Canonical pattern:

```go
transport := httpmock.NewMockTransport()
httpClient := &http.Client{Transport: transport}
client, err := NewFooClient(ctx,
    option.WithHTTPClient(httpClient),
    option.WithoutAuthentication(),
)
defer transport.Reset()

transport.RegisterResponder("GET", "https://api.example.com/resource/foo",
    func(req *http.Request) (*http.Response, error) {
        return httpmock.NewStringResponse(200, `{"name":"foo"}`), nil
    })
```

- Register responders on `transport` directly — not on the global `httpmock` state.
- Use `transport.Reset()` (not `httpmock.DeactivateAndReset()`) when using `NewMockTransport`.
- Tests must exercise real code paths: pagination loops, operation polling/wait, error propagation.
- Do not assert only that an error occurred; validate the specific behavior under test.

# Interface Mocks — use mockery, never hand-write

When introducing a new interface that wraps an external dependency:

1. Add a `mockery` entry to the `mocks` target in `Makefile`:
   ```makefile
   mockery --name=FooClientIntf --dir=internal/clients/foo --output=internal/clients/foo/mocks --outpkg=mocks
   ```
2. Run `make mocks` to generate the file.
3. Import and use the generated mock in all tests — never define a local struct that re-implements
   the interface inside a test file.

The generated mock in the `mocks` subpackage is the single source of truth for that interface.
Duplicate or hand-written mock implementations are not permitted.
