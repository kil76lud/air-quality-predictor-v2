def test_main_serves_ui_and_api(trained_champion, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("MODEL_DIR", str(trained_champion[0]))
    from aqi import api

    api.reset_state()
    import main

    c = TestClient(main.app)
    assert c.get("/health").json() == {"status": "ok"}
    assert c.get("/").status_code == 200  # UI de Gradio montada en la raíz
    assert c.get("/docs").status_code == 200
    api.reset_state()
