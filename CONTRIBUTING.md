# Contributing to Durbyn.jl

Thank you for your interest in contributing to Durbyn.jl! Durbyn is maintained
by the [Time Series Analysis and Forecasting Society (TAFS)](https://taf-society.org/),
a registered non-profit association under Austrian law, which ensures that all
contributions remain fully open source under the [Apache 2.0 license](LICENSE).

Everyone participating in this project is expected to follow our
[Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to Contribute

- **Report bugs** — file an issue with a reproducible example.
- **Improve documentation** — fix stale examples, clarify docstrings, add tutorials.
- **Add tests** — coverage of edge cases (integer inputs, missing values, short
  series, non-integer seasonal periods) is especially valuable.
- **Fix bugs or add features** — see the workflow below.
- **Port functionality** — Durbyn ports well-established forecasting methods
  (in the spirit of R's `forecast` package) to idiomatic Julia; ports of
  additional established methods are welcome.

## Reporting Bugs

A great bug report contains:

1. **A minimal reproducible example** — the smallest dataset and code that
   triggers the problem, e.g.:

   ```julia
   using Durbyn
   df = (ACTUAL = [1628, 1924, 1512, 1168, 1007],)
   spec = ArimaSpec(@formula(ACTUAL = p() + d() + q()))
   fit(spec, df, m = 7)   # ERROR: ...
   ```

2. **The full error message and stacktrace** — not just the first line.
3. **Your environment** — Julia version (`versioninfo()`) and Durbyn version
   (`] st Durbyn`).

Please check existing issues first to avoid duplicates.

## Development Setup

```bash
git clone https://github.com/taf-society/Durbyn.jl.git
cd Durbyn.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Durbyn supports Julia 1.10 and later.

## Running the Tests

Run the full test suite before opening a pull request:

```julia
using Pkg
Pkg.test()
```

Or run a single test file during development:

```julia
using Test, Durbyn
include("test/test_auto_arima.jl")
```

## Pull Request Workflow

1. **Fork** the repository and create a branch from `main`.
2. **Make focused changes** — one bug fix or feature per pull request.
3. **Add or update tests** that cover your change. A bug fix should include a
   regression test that fails without the fix.
4. **Keep docs in sync** — if you change a function signature, rename a keyword
   argument, or change exports, update the docstring **and** any examples in
   `README.md` and `docs/src/` that use it. Documentation examples are meant to
   be copy-paste runnable.
5. **Run the full test suite** and make sure it passes.
6. **Open the pull request** with a clear description of what changed and why.
   Link related issues (e.g. `Fixes #35`).

### Code Style

- Follow the existing style of the file you are editing.
- The project uses [JuliaFormatter](https://domluna.github.io/JuliaFormatter.jl/stable/)
  (configuration in `.JuliaFormatter.toml`).
- Prefer `Real` over concrete numeric types in public signatures so integer
  and float inputs both work; preserve `missing` where the API supports it.
- Keyword arguments that take a fixed set of options use `Symbol`s
  (e.g. `method = :css_ml`) validated with `_check_arg`.
- New public functions need a docstring with a runnable example, and should be
  exported from the owning module (and from `Durbyn` if part of the main API).
- When extending a generic from `Durbyn.Generics` (`fit`, `forecast`, `fitted`,
  `residuals`, `plot`, ...), remember to `import ..Generics: <name>` first —
  otherwise you create a private shadow function instead of a new method.

### Documentation

The documentation lives in `docs/` and is built with Documenter.jl:

```bash
julia --project=docs docs/make.jl
```

Code examples in the docs should run as written against the current API —
stale examples get copied by users and reported as bugs.

## Release Process (Maintainers)

1. Bump `version` in `Project.toml` (patch for fixes, minor for features).
2. Merge to `main`, then comment `@JuliaRegistrator register` on the release
   commit.
3. TagBot creates the git tag and GitHub release automatically after the
   registry PR merges.

## Questions?

Open a [GitHub issue](https://github.com/taf-society/Durbyn.jl/issues) or reach
out to TAFS at info@taf-society.org.
