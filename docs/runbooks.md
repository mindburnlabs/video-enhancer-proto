# Runbooks

## Warm Start

To preload models and validate the installation, run:

```bash
python warm_start.py
```

## Smoke Test

To run a sequence of API and UI invocations to ensure basic functionality:

```bash
pytest tests/test_smoke_e2e.py
```

## Rollback

To revert to a previous release, use git tags:

```bash
git checkout tags/<tag_name>
```
