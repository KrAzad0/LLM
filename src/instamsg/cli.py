"""Command line interface for the instant messaging engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .app import InstantMessagingApp, InstantMessagingError


def _load_app(path: Path | None) -> InstantMessagingApp:
    if path and path.exists():
        data = json.loads(path.read_text())
        return InstantMessagingApp.from_dict(data)
    return InstantMessagingApp()


def _save_app(path: Path | None, app: InstantMessagingApp) -> None:
    if path:
        path.write_text(json.dumps(app.to_dict(), indent=2))


def _print(message: str) -> None:
    print(message)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Instant messaging reference implementation")
    parser.add_argument("--state", type=Path, default=None, help="Path to a JSON file used to persist state")

    subparsers = parser.add_subparsers(dest="command", required=True)

    register_parser = subparsers.add_parser("register", help="Register a new user")
    register_parser.add_argument("username")
    register_parser.add_argument("--status", default="online")

    status_parser = subparsers.add_parser("status", help="Update a user's presence")
    status_parser.add_argument("username")
    status_parser.add_argument("status")

    channel_parser = subparsers.add_parser("channel", help="Channel management")
    channel_sub = channel_parser.add_subparsers(dest="channel_cmd", required=True)
    channel_create = channel_sub.add_parser("create", help="Create a channel")
    channel_create.add_argument("name")
    channel_create.add_argument("members", nargs="+", help="Usernames that should be part of the channel")
    channel_create.add_argument("--private", action="store_true")
    channel_create.add_argument("--topic", default=None)

    send_parser = subparsers.add_parser("send", help="Send a message to a channel")
    send_parser.add_argument("channel_id")
    send_parser.add_argument("sender")
    send_parser.add_argument("content")

    history_parser = subparsers.add_parser("history", help="Show channel history")
    history_parser.add_argument("channel_id")
    history_parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args(argv)
    state_path = args.state
    app = _load_app(state_path)

    try:
        if args.command == "register":
            user = app.register_user(args.username, status=args.status)
            _print(f"Registered {user.username} (id={user.id})")
        elif args.command == "status":
            user = app.get_user_by_username(args.username)
            app.set_user_status(user.id, args.status)
            _print(f"{user.username} is now {args.status}")
        elif args.command == "channel" and args.channel_cmd == "create":
            member_ids = [app.get_user_by_username(name).id for name in args.members]
            channel = app.create_channel(args.name, member_ids, is_private=args.private, topic=args.topic)
            _print(f"Created channel {channel.name} (id={channel.id})")
        elif args.command == "send":
            sender = app.get_user_by_username(args.sender)
            message = app.send_message(args.channel_id, sender.id, args.content)
            _print(f"[{message.timestamp:%H:%M}] {sender.username}: {message.content}")
        elif args.command == "history":
            messages = app.get_history(args.channel_id, limit=args.limit)
            for message in messages:
                author = app.get_user(message.sender_id)
                _print(f"[{message.timestamp:%Y-%m-%d %H:%M}] {author.username}: {message.content}")
        else:  # pragma: no cover - defensive
            raise InstantMessagingError("Unsupported command")
    except InstantMessagingError as exc:  # pragma: no cover - CLI surface
        _print(str(exc))
    else:
        _save_app(state_path, app)


if __name__ == "__main__":  # pragma: no cover
    main()
