from plumbing.secrets import require_anthropic_api_key, require_deepseek_api_key


def test_require_keys_loads_configured_secrets_file(monkeypatch, tmp_path) -> None:
    secrets = tmp_path / "secrets.env"
    secrets.write_text(
        "\n".join(
            [
                "# local test secrets",
                "ANTHROPIC_API_KEY=anthropic-secret",
                'export DEEPSEEK_API_KEY="deepseek-secret"',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HARNESS_SECRETS_FILE", str(secrets))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    assert require_anthropic_api_key() == "anthropic-secret"
    assert require_deepseek_api_key() == "deepseek-secret"
