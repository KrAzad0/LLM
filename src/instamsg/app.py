"""Core instant messaging engine."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Dict, Iterable, List, Optional

from .models import Channel, Message, User


class InstantMessagingError(RuntimeError):
    """Raised when an operation on the messaging engine fails."""


class InstantMessagingApp:
    """Simple in-memory instant messaging engine."""

    _allowed_statuses = {"online", "offline", "away", "busy"}

    def __init__(self) -> None:
        self._users: Dict[str, User] = {}
        self._channels: Dict[str, Channel] = {}
        self._messages: Dict[str, List[Message]] = defaultdict(list)
        self._read_state: Dict[str, Dict[str, Optional[datetime]]] = defaultdict(dict)
        self._username_index: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # User management
    # ------------------------------------------------------------------
    def register_user(self, username: str, *, status: str = "online") -> User:
        username = username.strip()
        if not username:
            raise InstantMessagingError("Username cannot be empty")
        key = username.lower()
        if key in self._username_index:
            raise InstantMessagingError(f"Username '{username}' is already taken")
        if status not in self._allowed_statuses:
            raise InstantMessagingError(f"Unsupported status '{status}'")
        user = User(username=username, status=status)
        self._users[user.id] = user
        self._username_index[key] = user.id
        return user

    def set_user_status(self, user_id: str, status: str) -> None:
        if status not in self._allowed_statuses:
            raise InstantMessagingError(f"Unsupported status '{status}'")
        user = self._require_user(user_id)
        user.status = status

    def add_contact(self, user_id: str, contact_id: str) -> None:
        user = self._require_user(user_id)
        contact = self._require_user(contact_id)
        if user_id == contact_id:
            raise InstantMessagingError("Cannot add yourself as a contact")
        user.contacts.add(contact_id)
        contact.contacts.add(user_id)

    def get_user(self, user_id: str) -> User:
        return self._require_user(user_id)

    def get_user_by_username(self, username: str) -> User:
        key = username.lower()
        if key not in self._username_index:
            raise InstantMessagingError(f"Unknown user '{username}'")
        return self._require_user(self._username_index[key])

    # ------------------------------------------------------------------
    # Channel management
    # ------------------------------------------------------------------
    def create_channel(
        self,
        name: str,
        member_ids: Iterable[str],
        *,
        is_private: bool = False,
        topic: str | None = None,
    ) -> Channel:
        clean_name = name.strip()
        if not clean_name:
            raise InstantMessagingError("Channel name cannot be empty")
        members = {self._require_user(member_id).id for member_id in member_ids}
        if len(members) < 2:
            raise InstantMessagingError("Channels require at least two members")
        channel = Channel(
            name=clean_name,
            member_ids=members,
            is_private=is_private,
            topic=topic,
        )
        self._channels[channel.id] = channel
        self._messages[channel.id] = []
        self._read_state[channel.id] = {member_id: None for member_id in members}
        return channel

    def list_channels(self, user_id: str) -> List[Channel]:
        self._require_user(user_id)
        return [channel for channel in self._channels.values() if user_id in channel.member_ids]

    def get_channel(self, channel_id: str) -> Channel:
        return self._require_channel(channel_id)

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------
    def send_message(self, channel_id: str, sender_id: str, content: str) -> Message:
        channel = self._require_channel(channel_id)
        self._require_user(sender_id)
        if sender_id not in channel.member_ids:
            raise InstantMessagingError("Sender is not a member of this channel")
        clean_content = content.strip()
        if not clean_content:
            raise InstantMessagingError("Messages cannot be empty")
        message = Message(channel_id=channel_id, sender_id=sender_id, content=clean_content)
        self._messages[channel_id].append(message)
        channel.last_message_at = message.timestamp
        self._read_state[channel_id][sender_id] = message.timestamp
        return message

    def get_history(self, channel_id: str, *, limit: Optional[int] = None) -> List[Message]:
        history = list(self._messages[self._require_channel(channel_id).id])
        if limit is not None:
            return history[-limit:]
        return history

    def mark_read(self, channel_id: str, user_id: str) -> None:
        channel = self._require_channel(channel_id)
        self._require_user(user_id)
        if user_id not in channel.member_ids:
            raise InstantMessagingError("User is not a member of the channel")
        messages = self._messages[channel_id]
        if messages:
            self._read_state[channel_id][user_id] = messages[-1].timestamp

    def get_unread_count(self, channel_id: str, user_id: str) -> int:
        channel = self._require_channel(channel_id)
        self._require_user(user_id)
        if user_id not in channel.member_ids:
            raise InstantMessagingError("User is not a member of the channel")
        last_read = self._read_state[channel_id].get(user_id)
        if last_read is None:
            return len(self._messages[channel_id])
        return sum(1 for message in self._messages[channel_id] if message.timestamp > last_read)

    def search_messages(
        self,
        query: str,
        *,
        channel_id: str | None = None,
        sender_id: str | None = None,
    ) -> List[Message]:
        search = query.strip().lower()
        if not search:
            raise InstantMessagingError("Search query cannot be empty")
        channels = [channel_id] if channel_id else list(self._channels.keys())
        results: List[Message] = []
        for cid in channels:
            for message in self._messages.get(cid, []):
                if sender_id and message.sender_id != sender_id:
                    continue
                if search in message.content.lower():
                    results.append(message)
        return results

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, object]:
        return {
            "users": [
                {
                    "id": user.id,
                    "username": user.username,
                    "status": user.status,
                    "contacts": sorted(user.contacts),
                }
                for user in self._users.values()
            ],
            "channels": [
                {
                    "id": channel.id,
                    "name": channel.name,
                    "member_ids": sorted(channel.member_ids),
                    "is_private": channel.is_private,
                    "topic": channel.topic,
                    "created_at": channel.created_at.isoformat(),
                    "last_message_at": channel.last_message_at.isoformat()
                    if channel.last_message_at
                    else None,
                }
                for channel in self._channels.values()
            ],
            "messages": {
                channel_id: [
                    {
                        "id": message.id,
                        "channel_id": message.channel_id,
                        "sender_id": message.sender_id,
                        "content": message.content,
                        "timestamp": message.timestamp.isoformat(),
                    }
                    for message in self._messages[channel_id]
                ]
                for channel_id in self._messages
            },
            "read_state": {
                channel_id: {user_id: ts.isoformat() if ts else None for user_id, ts in readers.items()}
                for channel_id, readers in self._read_state.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "InstantMessagingApp":
        app = cls()
        for user_data in payload.get("users", []):
            user = User(
                username=user_data["username"],
                id=user_data["id"],
                status=user_data.get("status", "offline"),
                contacts=set(user_data.get("contacts", [])),
            )
            app._users[user.id] = user
            app._username_index[user.username.lower()] = user.id
        for channel_data in payload.get("channels", []):
            channel = Channel(
                name=channel_data["name"],
                member_ids=set(channel_data.get("member_ids", [])),
                is_private=channel_data.get("is_private", False),
                topic=channel_data.get("topic"),
                id=channel_data["id"],
                created_at=datetime.fromisoformat(channel_data["created_at"]),
                last_message_at=
                    datetime.fromisoformat(channel_data["last_message_at"])
                    if channel_data.get("last_message_at")
                    else None,
            )
            app._channels[channel.id] = channel
        for channel_id, message_list in payload.get("messages", {}).items():
            restored: List[Message] = []
            for message_data in message_list:
                restored.append(
                    Message(
                        channel_id=message_data["channel_id"],
                        sender_id=message_data["sender_id"],
                        content=message_data["content"],
                        timestamp=datetime.fromisoformat(message_data["timestamp"]),
                        id=message_data["id"],
                    )
                )
            app._messages[channel_id] = restored
        for channel_id, readers in payload.get("read_state", {}).items():
            app._read_state[channel_id] = {
                user_id: datetime.fromisoformat(ts) if ts else None for user_id, ts in readers.items()
            }
        return app

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _require_user(self, user_id: str) -> User:
        if user_id not in self._users:
            raise InstantMessagingError(f"Unknown user '{user_id}'")
        return self._users[user_id]

    def _require_channel(self, channel_id: str) -> Channel:
        if channel_id not in self._channels:
            raise InstantMessagingError(f"Unknown channel '{channel_id}'")
        return self._channels[channel_id]


__all__ = ["InstantMessagingApp", "InstantMessagingError"]
