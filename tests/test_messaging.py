"""Tests for the RelayChat instant messaging engine."""

from instamsg import InstantMessagingApp, InstantMessagingError


def _bootstrap_app():
    app = InstantMessagingApp()
    alice = app.register_user("alice")
    bob = app.register_user("bob", status="away")
    carol = app.register_user("carol")
    channel = app.create_channel("general", [alice.id, bob.id, carol.id])
    return app, alice, bob, carol, channel


def test_user_registration_and_status():
    app = InstantMessagingApp()
    alice = app.register_user("Alice", status="busy")
    assert app.get_user(alice.id).status == "busy"
    app.set_user_status(alice.id, "away")
    assert app.get_user(alice.id).status == "away"
    try:
        app.register_user("alice")
    except InstantMessagingError:
        pass
    else:  # pragma: no cover
        raise AssertionError("Expected InstantMessagingError")


def test_channel_message_flow_and_unread_counts():
    app, alice, bob, carol, channel = _bootstrap_app()
    message = app.send_message(channel.id, alice.id, "Hello team!")
    assert message.content == "Hello team!"
    assert app.get_unread_count(channel.id, bob.id) == 1
    app.mark_read(channel.id, bob.id)
    assert app.get_unread_count(channel.id, bob.id) == 0
    app.send_message(channel.id, carol.id, "Lunch at 1?")
    history = app.get_history(channel.id)
    assert [msg.content for msg in history][-1] == "Lunch at 1?"


def test_search_and_contacts():
    app, alice, bob, carol, channel = _bootstrap_app()
    app.add_contact(alice.id, bob.id)
    assert bob.id in app.get_user(alice.id).contacts
    app.send_message(channel.id, alice.id, "Deploying the update now")
    app.send_message(channel.id, bob.id, "Copy that")
    results = app.search_messages("update")
    assert len(results) == 1
    assert results[0].sender_id == alice.id


def test_serialization_round_trip():
    app, alice, bob, _, channel = _bootstrap_app()
    app.send_message(channel.id, alice.id, "Initial ping")
    app.mark_read(channel.id, bob.id)
    payload = app.to_dict()
    restored = InstantMessagingApp.from_dict(payload)
    restored_history = restored.get_history(channel.id)
    assert [msg.content for msg in restored_history] == ["Initial ping"]
    assert restored.get_unread_count(channel.id, bob.id) == 0
