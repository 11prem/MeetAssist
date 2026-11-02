import pytest

from meetassist import paths


@pytest.fixture(autouse=True, scope="session")
def _sandbox_user_data(tmp_path_factory):
    """Point every user-data path at a session temp dir. Per-test monkeypatches restore to THIS, never to the real
    %LOCALAPPDATA%\\MeetAssist - a closed window's timer or a thread outliving its test must never write test data there."""
    root = tmp_path_factory.mktemp("userdata")
    sandbox = {"DATA_DIR": root, "MEETINGS_DIR": root / "meetings", "SETTINGS_FILE": root / "settings.json",
               "LOG_FILE": root / "app.log"}
    with pytest.MonkeyPatch.context() as mp:
        for name, value in sandbox.items():
            mp.setattr(paths, name, value)
        yield
