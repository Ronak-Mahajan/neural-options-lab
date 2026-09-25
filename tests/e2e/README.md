# Browser acceptance checks

`verify_ux.py` drives the dashboard and the methodology page in headless Chromium
and prints one `PASS` or `FAIL` line per check, ending with `ALL UX CHECKS PASSED`
when every check passes (exit code 0) or with the failing checks (exit code 1). It
covers the quote card and model card, the hedging run, the desk note, the
short-dated regime, the help bubbles, URL state and the layout at phone and
desktop widths. The file name has no `test_` prefix, so `pytest` does not collect
it; it needs a running server.

## Run it against a local server

```bash
python -m pip install playwright
python -m playwright install chromium

# terminal 1: the app
python -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8090

# terminal 2: the checks
python tests/e2e/verify_ux.py http://127.0.0.1:8090
```

An optional second argument names a directory for screenshots:
`python tests/e2e/verify_ux.py http://127.0.0.1:8090 ux-shots`.

The first price takes a few seconds while the server loads its checkpoints, and the
hedging run takes longer; the script waits up to three minutes for a price and five
for the hedging run.

## The Content-Security-Policy

The server sends an enforcing Content-Security-Policy with no `unsafe-eval`.
Playwright's `wait_for_function` evaluates its string predicates with `eval`, so
the script opens every browser context with `bypass_csp=True`. That exemption is
for the test harness only. To check the page itself under the policy, load it in a
context without `bypass_csp` and count `securitypolicyviolation` events and console
errors.
