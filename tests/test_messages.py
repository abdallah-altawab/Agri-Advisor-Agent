from agri_advisor.messages import content_to_text


def test_plain_string():
    assert content_to_text("hello") == "hello"


def test_text_blocks_ignore_extras():
    blocks = [
        {"type": "text", "text": "foo"},
        {"type": "text", "text": "bar", "extras": {"signature": "xyz"}},
    ]
    assert content_to_text(blocks) == "foobar"


def test_skips_non_text_blocks():
    blocks = [{"type": "image", "data": "..."}, {"type": "text", "text": "ok"}]
    assert content_to_text(blocks) == "ok"


def test_none_and_empty():
    assert content_to_text(None) == ""
    assert content_to_text([]) == ""