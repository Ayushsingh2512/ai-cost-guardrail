from app.core.config import Settings
from app.main import create_app


def _has_token_route(app):
    return "/token" in app.openapi()["paths"]


def _has_tenants_route(app):
    return "/tenants/" in app.openapi()["paths"]


def test_token_route_exists_in_development():
    app = create_app(
        Settings(
            environment="development",
            gemini_api_key="test-key",
            jwt_secret="test-secret",
        )
    )

    assert _has_token_route(app) is True


def test_token_route_does_not_exist_in_production():
    app = create_app(
        Settings(
            environment="production",
            gemini_api_key="test-key",
            jwt_secret="test-secret",
        )
    )

    assert _has_token_route(app) is False


def test_tenants_route_exists_in_development():
    app = create_app(
        Settings(
            environment="development",
            gemini_api_key="test-key",
            jwt_secret="test-secret",
        )
    )

    assert _has_tenants_route(app) is True


def test_tenants_route_does_not_exist_in_production():
    app = create_app(
        Settings(
            environment="production",
            gemini_api_key="test-key",
            jwt_secret="test-secret",
        )
    )

    assert _has_tenants_route(app) is False