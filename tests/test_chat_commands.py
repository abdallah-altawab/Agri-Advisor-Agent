import pytest

from agri_advisor.chat_commands import ImageCommand, parse_image_command, with_attachment


def test_plain_text_is_not_a_command():
    assert parse_image_command("hello there") is None
    assert parse_image_command("/imagery is nice") is None


def test_unquoted_path_and_text():
    command = parse_image_command("/image leaf.jpg what is this?")
    assert command == ImageCommand("leaf.jpg", "what is this?")


def test_quoted_path_with_spaces():
    command = parse_image_command('/image "C:\\My Photos\\leaf 1.jpg" هل ده مرض؟')
    assert command == ImageCommand(r"C:\My Photos\leaf 1.jpg", "هل ده مرض؟")


def test_bad_usage_raises():
    with pytest.raises(ValueError, match="Usage"):
        parse_image_command("/image")
    with pytest.raises(ValueError, match="quote"):
        parse_image_command('/image "C:\\oops.jpg')


def test_with_attachment_adds_note_and_default_text():
    assert with_attachment("look", "abc.jpg") == "look\n[image attached: abc.jpg]"
    message = with_attachment("", "abc.jpg")
    assert message.startswith("Please look")
    assert message.endswith("[image attached: abc.jpg]")