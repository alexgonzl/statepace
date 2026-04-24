# tests/ — organization schema

Governance doc for `tests/`. Follow this layout for every new test file, fixture, or helper.

```
tests/
├── README.md               ← this file
├── __init__.py             ← package marker; keep empty
├── test_<module>.py        ← tests for statepace/<module>.py
├── test_<package>_<name>.py ← tests for statepace/<package>/<name>.py (e.g. test_evaluation_harness.py)
├── conftest.py             ← pytest fixtures (only when justified; see rules)
└── fixtures/
    ├── __init__.py         ← package marker; keep empty
    ├── <purpose>.py        ← plain factory functions (e.g. synthetic.py)
    └── reference_impls/    ← spec-scoped fixtures; governed by `reference_impls/README.md`
```

## Rules

1. **Mirror the package layout.** Test file for `statepace/X.py` is `tests/test_X.py`. Test file for `statepace/pkg/Y.py` is `tests/test_pkg_Y.py`. No milestone numbers, no feature names — tests are permanent; plans are not.
2. **Fixtures are plain factory functions** under `tests/fixtures/`. Take explicit arguments (no defaults), return the object. No RNG state held internally; callers pass seeded RNGs if randomness is needed. No side effects (no disk reads, no network). Usable outside pytest.
3. **`conftest.py` is for pytest wiring only** — fixtures registered as `@pytest.fixture`, session-scoped setup, plugin hooks. It must not contain factory logic; it wraps factories from `tests/fixtures/`. Add `conftest.py` only when a pytest fixture is genuinely needed; don't create it preemptively.
4. **No production code in `tests/`.** Anything that would be imported from `statepace/` belongs in `statepace/`. Tests import from the package; the package does not import from tests.
5. **No shared mutable state between tests.** Each test constructs what it needs from the factory. No module-level fixtures holding arrays.
6. **Real fixtures, no mocking internals** (per CLAUDE.md). If a test needs a `Channels`, build one via `tests/fixtures/synthetic.py`; don't mock the dataclass.

## Adding something new

| New thing | Where it goes |
|---|---|
| Test for `statepace/foo.py` | `tests/test_foo.py` |
| Test for `statepace/pkg/bar.py` | `tests/test_pkg_bar.py` |
| Factory function (synthetic `Channels`, synthetic `Prior`, etc.) | `tests/fixtures/<purpose>.py` |
| Spec-scoped fixture for a reference impl | `tests/fixtures/reference_impls/<slug>.py` |
| pytest-specific fixture wrapping a factory | `tests/conftest.py` |
| Integration test spanning multiple modules | `tests/test_integration_<scope>.py` |

If a new artifact doesn't fit the table, extend this README before adding it.
