"""Provider requests are mocked; no credentials, billable requests, or login prompts."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from quibble import planner as foundry
from quibble.generate import recommendation


@pytest.fixture
def configured(monkeypatch):
    for name in ("LLM_BASE_URL", "LLM_MODEL", "LLM_AUTH_MODE", "LLM_API_KEY",
                 "LLM_AAD_CREDENTIAL", "LLM_TOKEN_SCOPE", "LLM_PROVIDER_NAME",
                 "OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL"):
        monkeypatch.setenv(name, "")
    monkeypatch.setenv("FOUNDRY_PROJECT_ENDPOINT", "https://example.openai.azure.com/openai/v1/")
    monkeypatch.setenv("AZURE_AI_MODEL_DEPLOYMENT_NAME", "test-deployment")
    monkeypatch.setenv("FOUNDRY_AUTH_MODE", "key")
    monkeypatch.setenv("FOUNDRY_API_KEY", "fake-test-key")
    monkeypatch.setenv("FOUNDRY_AAD_CREDENTIAL", "cli")
    monkeypatch.setenv("AZURE_TENANT_ID", "")
    monkeypatch.setenv("AZURE_SUBSCRIPTION_ID", "")
    monkeypatch.setenv("AZURE_OPENAI_SCOPE", "https://ai.azure.com/.default")


@pytest.mark.parametrize("endpoint", ["http://remote.example/openai/v1", "https://x/api/projects/y",
                                     "https://user:password@x/openai/v1", "https://x/openai/v1?api-key=secret"])
def test_invalid_endpoint(configured, monkeypatch, endpoint):
    monkeypatch.setenv("FOUNDRY_PROJECT_ENDPOINT", endpoint)
    with pytest.raises(foundry.PlannerError, match="inference base URL"):
        foundry.Settings.from_env()


def fake_client(monkeypatch, response):
    captured = {}
    class Client:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.chat = SimpleNamespace(completions=self)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def create(self, **kwargs):
            captured["request"] = kwargs
            return response
    monkeypatch.setattr("openai.OpenAI", Client)
    return captured


def reply(spec, finish="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish,
                           message=SimpleNamespace(content=json.dumps(spec)))],
                           model="reported-model", id="response-test", usage=None)


def test_plan_only_sends_scenario_and_controls(configured, monkeypatch):
    spec = recommendation(rows=100, fields=3)
    captured = fake_client(monkeypatch, reply(spec))
    result = foundry.plan("A fictional farm predicts yield.", 100, 3, 42, "classification")
    assert result["spec"]["scenario"] == "A fictional farm predicts yield."
    assert captured["client"]["api_key"] == "fake-test-key"
    assert captured["client"]["max_retries"] == 0
    request = captured["request"]
    assert request["model"] == "test-deployment"
    assert request["response_format"] == {"type": "json_object"}
    assert len(request["messages"]) == 2
    assert set(json.loads(request["messages"][1]["content"])) == {
        "scenario", "rows", "predictors", "seed", "task", "example_specification"}
    assert "fake-test-key" not in json.dumps(result)


def test_regression_prompt_requires_explicit_reinterpretation(configured, monkeypatch):
    spec = recommendation(rows=100, fields=3, task="regression")
    captured = fake_client(monkeypatch, reply(spec))
    result = foundry.plan("Will clothing inventory cover demand before delivery?",
                         100, 3, 42, "regression")
    prompt = captured["request"]["messages"][0]["content"]
    assert "selected task takes precedence" in prompt
    assert "reinterpretation" in prompt and "continuous synthetic score" in prompt
    assert result["spec"]["task"] == "regression"
    assert result["spec"]["target"] == "inventory_pressure"


@pytest.mark.parametrize("problem", ["shape", "code", "truncated"])
def test_invalid_model_output_is_not_used(configured, monkeypatch, problem):
    spec = recommendation(rows=100, fields=3)
    if problem == "shape":
        spec["rows"] = 101
    if problem == "code":
        spec["python"] = "raise Exception()"
    fake_client(monkeypatch, reply(spec, finish="length" if problem == "truncated" else "stop"))
    with pytest.raises(foundry.PlannerError):
        foundry.plan("fictional", 100, 3, 42, "classification")


def test_invalid_interaction_gets_one_schema_guided_correction(configured, monkeypatch):
    invalid = recommendation(rows=100, fields=3)
    invalid["interactions"] = [{"features": ["inventory_coverage", "delivery_lead_days"], "weight": 0.5}]
    corrected = recommendation(rows=100, fields=3)
    corrected["interactions"] = [
        {"left": "inventory_coverage", "right": "delivery_lead_days", "weight": 0.5}]
    replies = iter([reply(invalid), reply(corrected)])
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return next(replies)

    fake_client(monkeypatch, reply(invalid))
    import openai
    client_type = openai.OpenAI

    class Client(client_type):
        def create(self, **kwargs):
            return create(**kwargs)

    monkeypatch.setattr(openai, "OpenAI", Client)
    result = foundry.plan("A fictional farm predicts yield.", 100, 3, 42, "classification")
    assert len(requests) == 2
    assert len(requests[0]["messages"]) == 2 and len(requests[1]["messages"]) == 4
    assert "interaction requires exactly: left right weight" in requests[1]["messages"][-1]["content"]
    assert result["spec"]["interactions"] == corrected["interactions"]
    assert result["planner"]["validation_correction_attempts"] == 1
    assert '"left":"predictor_name"' in requests[0]["messages"][0]["content"]


def test_invalid_interactions_stop_after_one_correction(configured, monkeypatch):
    invalid = recommendation(rows=100, fields=3)
    invalid["interactions"] = [{"left": "inventory_coverage", "weight": 1}]
    captured = fake_client(monkeypatch, reply(invalid))
    with pytest.raises(foundry.PlannerError, match="interaction requires exactly"):
        foundry.plan("fictional", 100, 3, 42, "classification")
    assert len(captured["request"]["messages"]) == 4


def test_aad_provider_and_scope(configured, monkeypatch):
    monkeypatch.setenv("FOUNDRY_AUTH_MODE", "aad")
    credential = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(foundry, "credential_for", lambda settings: credential)
    captured_scope = []
    def provider(c, scope):
        assert c is credential
        captured_scope.append(scope)
        return lambda: "fake-token"
    monkeypatch.setattr("azure.identity.get_bearer_token_provider", provider)
    captured = fake_client(monkeypatch, reply(recommendation(rows=100, fields=3)))
    foundry.plan("fictional", 100, 3, 42, "classification")
    assert captured_scope == ["https://ai.azure.com/.default"]
    assert callable(captured["client"]["api_key"])


def test_provider_error_body_is_not_exposed(configured, monkeypatch):
    import openai
    fake_client(monkeypatch, reply(recommendation()))
    class Client:
        def __init__(self, **kwargs):
            raise openai.APIError("fake-test-key and private response", request=None, body=None)
    monkeypatch.setattr(openai, "OpenAI", Client)
    with pytest.raises(foundry.PlannerError) as caught:
        foundry.plan("fictional", 100, 3, 42, "classification")
    assert "fake-test-key" not in str(caught.value)


def test_explicit_login_and_token_check(configured, monkeypatch, capsys):
    monkeypatch.setenv("FOUNDRY_AUTH_MODE", "aad")
    monkeypatch.setenv("AZURE_TENANT_ID", "test-tenant")
    monkeypatch.setenv("AZURE_SUBSCRIPTION_ID", "test-subscription")
    monkeypatch.setattr(foundry.shutil, "which", lambda n: "az")
    commands = []
    monkeypatch.setattr(foundry.subprocess, "run", lambda command, **kwargs: commands.append(command))
    monkeypatch.setattr(foundry, "credential_for", lambda s: SimpleNamespace(
        get_token=lambda scope: SimpleNamespace(token="never-print-this"), close=lambda: None))
    monkeypatch.setattr("sys.argv", ["quibble auth", "--device-code"])
    assert foundry.main() == 0
    assert commands[0] == ["az", "login", "--output", "none", "--tenant", "test-tenant", "--use-device-code"]
    assert commands[1] == ["az", "account", "set", "--subscription", "test-subscription"]
    assert "never-print-this" not in capsys.readouterr().out
    commands.clear()
    monkeypatch.setattr("sys.argv", ["quibble auth", "--check"])
    assert foundry.main() == 0 and not commands


def test_default_identity_does_not_trigger_cli_login(configured, monkeypatch):
    monkeypatch.setenv("FOUNDRY_AUTH_MODE", "aad")
    monkeypatch.setenv("FOUNDRY_AAD_CREDENTIAL", "default")
    monkeypatch.setattr("sys.argv", ["quibble auth"])
    def forbidden(*args, **kwargs):
        pytest.fail("Default identity must not invoke CLI login")
    monkeypatch.setattr(foundry.subprocess, "run", forbidden)
    assert foundry.main() == 1


def test_invalid_credential_initialization_is_explicit(configured, monkeypatch, capsys):
    monkeypatch.setenv("FOUNDRY_AUTH_MODE", "aad")
    monkeypatch.setattr("sys.argv", ["quibble auth", "--check"])
    def bad_credential(settings):
        raise ValueError("private credential details")
    monkeypatch.setattr(foundry, "credential_for", bad_credential)
    assert foundry.main() == 1
    assert "private credential details" not in capsys.readouterr().err


def test_openai_standard_environment_without_azure(configured, monkeypatch):
    monkeypatch.setenv("FOUNDRY_PROJECT_ENDPOINT", "")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
    monkeypatch.setenv("OPENAI_MODEL", "my-openai-model")
    settings = foundry.Settings.from_env()
    assert settings.endpoint == "https://api.openai.com/v1/"
    assert settings.mode == "api_key" and settings.model == "my-openai-model"
    captured = fake_client(monkeypatch, reply(recommendation(rows=100, fields=3)))
    result = foundry.plan("fictional", 100, 3, 42, "classification")
    assert captured["client"]["api_key"] == "fake-openai-key"
    assert result["planner"]["model"] == "my-openai-model"
    assert result["planner"]["provider"] == "api.openai.com"


def test_user_chosen_compatible_provider(configured, monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://inference.example/api/v1")
    monkeypatch.setenv("LLM_MODEL", "chosen-model")
    monkeypatch.setenv("LLM_AUTH_MODE", "api_key")
    monkeypatch.setenv("LLM_API_KEY", "provider-key")
    monkeypatch.setenv("LLM_PROVIDER_NAME", "My provider")
    captured = fake_client(monkeypatch, reply(recommendation(rows=100, fields=3)))
    result = foundry.plan("fictional", 100, 3, 42, "classification")
    assert captured["client"]["base_url"] == "https://inference.example/api/v1/"
    assert captured["client"]["api_key"] == "provider-key"
    assert captured["request"]["model"] == "chosen-model"
    assert result["planner"]["provider"] == "My provider"


def test_custom_provider_never_implicitly_gets_openai_or_azure_key(configured, monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://other.example/v1")
    monkeypatch.setenv("LLM_MODEL", "other-model")
    monkeypatch.setenv("OPENAI_API_KEY", "private-openai-key")
    with pytest.raises(foundry.PlannerError, match="required"):
        foundry.Settings.from_env()


@pytest.mark.parametrize("endpoint", ["http://localhost:11434/v1", "http://127.0.0.1:8080/v1",
                                     "http://[::1]:8080/v1"])
def test_local_service_without_real_credentials(configured, monkeypatch, endpoint):
    monkeypatch.setenv("LLM_BASE_URL", endpoint)
    monkeypatch.setenv("LLM_MODEL", "local-model")
    monkeypatch.setenv("LLM_AUTH_MODE", "none")
    captured = fake_client(monkeypatch, reply(recommendation(rows=100, fields=3)))
    foundry.plan("fictional", 100, 3, 42, "classification")
    assert captured["client"]["api_key"] == "local-no-auth"
    assert "fake-test-key" not in repr(captured)


@pytest.mark.parametrize("endpoint,auth", [("http://public.example/v1", "api_key"),
                                         ("https://public.example/v1", "none"),
                                         ("http://127.0.0.1/v1", "aad")])
def test_invalid_transport_and_auth_combinations(configured, monkeypatch, endpoint, auth):
    monkeypatch.setenv("LLM_BASE_URL", endpoint)
    monkeypatch.setenv("LLM_MODEL", "model")
    monkeypatch.setenv("LLM_AUTH_MODE", auth)
    with pytest.raises(foundry.PlannerError):
        foundry.Settings.from_env()


def test_generic_azure_aad_settings(configured, monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://example.openai.azure.com/openai/v1")
    monkeypatch.setenv("LLM_MODEL", "azure-deployment")
    monkeypatch.setenv("LLM_AUTH_MODE", "aad")
    monkeypatch.setenv("LLM_AAD_CREDENTIAL", "default")
    monkeypatch.setenv("LLM_TOKEN_SCOPE", "https://ai.azure.com/.default")
    settings = foundry.Settings.from_env()
    assert settings.mode == "aad" and settings.credential == "default"
    assert settings.model == "azure-deployment"


def test_openai_key_does_not_require_azure_sdk(configured, monkeypatch):
    import builtins
    original_import = builtins.__import__
    def without_azure(name, *args, **kwargs):
        if name == "azure" or name.startswith("azure."):
            raise ImportError("Azure SDK intentionally unavailable")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", without_azure)
    fake_client(monkeypatch, reply(recommendation(rows=100, fields=3)))
    assert foundry.plan("fictional", 100, 3, 42, "classification")["spec"]["rows"] == 100


@pytest.mark.parametrize("auth", ["api_key", "none"])
def test_real_sdk_against_compatible_http_endpoint(configured, monkeypatch, auth):
    observed = {}
    spec = recommendation(rows=100, fields=3)
    class Endpoint(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            observed["path"] = self.path
            observed["authorization"] = self.headers.get("Authorization")
            observed["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            payload = json.dumps({
                "id": "local-response", "object": "chat.completion", "created": 1,
                "model": "reported-local-model",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": json.dumps(spec)}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Endpoint)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    monkeypatch.setenv("LLM_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.setenv("LLM_MODEL", "chosen-local-model")
    monkeypatch.setenv("LLM_AUTH_MODE", auth)
    monkeypatch.setenv("LLM_API_KEY", "local-test-key")
    try:
        result = foundry.plan("A fictional farm predicts yield.", 100, 3, 42, "classification")
        assert observed["path"] == "/v1/chat/completions"
        assert observed["body"]["model"] == "chosen-local-model"
        assert observed["authorization"] == f"Bearer {'local-test-key' if auth == 'api_key' else 'local-no-auth'}"
        assert result["planner"]["usage"]["total_tokens"] == 30
        assert result["planner"]["reported_model"] == "reported-local-model"
        assert result["spec"]["scenario"] == "A fictional farm predicts yield."
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
