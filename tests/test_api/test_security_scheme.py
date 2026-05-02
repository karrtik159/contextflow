from app.api.deps import oauth2_scheme


def test_oauth2_token_url_matches_registered_login_route():
    assert oauth2_scheme.model.flows.password.tokenUrl == "/api/v1/login"
