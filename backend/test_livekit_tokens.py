"""LiveKit room naming and URL helpers. Does not call the LiveKit network."""

from interviewer.services.livekit_tokens import livekit_client_url, mint_agent_token, room_name_for_session


def test_room_is_one_session():
    assert room_name_for_session("sess-1") == "interview-sess-1"
    assert room_name_for_session("sess-2") == "interview-sess-2"


def test_https_dashboard_url_becomes_wss():
    assert livekit_client_url("https://proj.livekit.cloud") == "wss://proj.livekit.cloud"
    assert livekit_client_url("wss://proj.livekit.cloud") == "wss://proj.livekit.cloud"


def test_agent_room_matches_candidate_session():
    from interviewer.services.livekit_tokens import is_livekit_configured

    if not is_livekit_configured():
        return
    sid = "sess-phase2"
    token = mint_agent_token(sid)
    assert token["room"] == room_name_for_session(sid)
    assert token["identity"].startswith("ai-interviewer-")
    assert token["token"]
    assert token["url"].startswith("ws")
