import sqlite3

import pytest

from verireason.demo import prepare_demo_prompt


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize and add_generation_prompt and not enable_thinking
        text = "\n".join(item["content"] for item in messages)
        return list(range(max(1, len(text) // 8)))


def _config(maximum_prompt_tokens=2000):
    return {
        "prompt": {
            "system": "Return SQL only.",
            "max_sequence_tokens": maximum_prompt_tokens + 10,
        },
        "decoding": {"max_new_tokens": 10},
    }


def test_prepare_demo_prompt_uses_read_only_schema(tmp_path):
    database = tmp_path / "shop.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE products(id INTEGER PRIMARY KEY, revenue REAL NOT NULL)")
    connection.commit()
    connection.close()

    prompt = prepare_demo_prompt(
        database,
        "Which product has the most revenue?",
        FakeTokenizer(),
        _config(),
    )

    assert 'Table "products"' in prompt["schema"]
    assert "Question:" in prompt["messages"][1]["content"]
    assert prompt["dropped_tables"] == []


def test_prepare_demo_prompt_rejects_empty_question(tmp_path):
    database = tmp_path / "shop.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE products(id INTEGER)")
    connection.commit()
    connection.close()

    with pytest.raises(ValueError, match="question cannot be empty"):
        prepare_demo_prompt(database, " ", FakeTokenizer(), _config())


def test_prepare_demo_prompt_drops_only_whole_trailing_tables(tmp_path):
    database = tmp_path / "shop.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE a(id INTEGER)")
    connection.execute("CREATE TABLE z(extra_long_column_name TEXT)")
    connection.commit()
    connection.close()

    prompt = prepare_demo_prompt(database, "List ids", FakeTokenizer(), _config(16))

    assert prompt["dropped_tables"] == ["z"]
    assert 'Table "a"' in prompt["schema"]
    assert 'Table "z"' not in prompt["schema"]
