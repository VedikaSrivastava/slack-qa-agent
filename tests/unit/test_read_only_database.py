import sqlite3
from pathlib import Path

import pytest

from slack_qa_agent.retrieval.database import open_read_only_database


def test_knowledge_database_rejects_writes(tmp_path: Path) -> None:
    path = tmp_path / "knowledge.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE artifacts (id TEXT PRIMARY KEY, title TEXT, content TEXT)")
    connection.commit()
    connection.close()

    read_only = open_read_only_database(path)
    with pytest.raises(sqlite3.OperationalError):
        read_only.execute("INSERT INTO artifacts VALUES ('1', 'title', 'body')")
    read_only.close()
