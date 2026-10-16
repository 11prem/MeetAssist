# AGENTS.md — standing instructions for coding agents

## Hard rules (non-negotiable)

1. Capture must **never** block on transcription.
2. **Never delete raw audio** until the user explicitly closes a meeting.
3. **Never** make a network call except to the LLM endpoint configured in Settings.
4. **Never overwrite** raw transcript text with LLM-corrected text. Store both.
5. **No admin rights** — nothing needing a driver, a service, or a system-wide install.

## Environment

- Target Python 3.11, Windows, x86-64.
- Install from PyPI (`pip install -r requirements.txt`). If your network requires a private
  mirror/proxy, point `pip`/`npm` at it via the usual `--index-url` / `.npmrc` config rather than
  hardcoding it here.
- Type hints throughout; `pytest` for tests.
- Check that any new runtime dependency actually resolves in your environment before adding it.
- **Never commit** anything under `models/` or `data/`.
