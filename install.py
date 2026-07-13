from analyzer_installer import ensure_analyzer_sync

if __name__ == "__main__":
    result = ensure_analyzer_sync(force=False)
    if result.get("state") != "ready":
        raise SystemExit(result.get("message", "Analyzer installation failed."))
