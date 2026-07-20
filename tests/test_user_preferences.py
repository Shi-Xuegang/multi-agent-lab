"""用户偏好序列化测试。"""

import json
from datetime import datetime
from pathlib import Path

from src.memory.user_preferences import UserPreferenceManager


def test_save_with_datetime(tmp_path: Path):
    path = tmp_path / "prefs.json"
    mgr = UserPreferenceManager(storage_path=str(path))
    result = mgr.update_preferences(
        {
            "username": "u1",
            "deadline": datetime(2026, 1, 1, 12, 0, 0),
            "research_history": ["主题A"],
        }
    )
    assert result.success
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["username"] == "u1"
    assert "2026" in data["deadline"]


def test_add_research_topic(tmp_path: Path):
    path = tmp_path / "prefs2.json"
    mgr = UserPreferenceManager(storage_path=str(path))
    mgr.add_research_topic("主题一")
    mgr.add_research_topic("主题二")
    mgr.add_research_topic("主题一")
    prefs = mgr.get_preferences()
    assert prefs.research_history[-1] == "主题一"
    assert prefs.research_history.count("主题一") == 1
