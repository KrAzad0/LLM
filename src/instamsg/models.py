"""Data structures used by the instant messaging engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import uuid
from typing import Set


@dataclass(slots=True)
class User:
    """Representation of a registered user."""

    username: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: str = "offline"
    contacts: Set[str] = field(default_factory=set)


@dataclass(slots=True)
class Channel:
    """Group or direct message channel."""

    name: str
    member_ids: Set[str]
    is_private: bool = False
    topic: str | None = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_message_at: datetime | None = None


@dataclass(slots=True)
class Message:
    """A message posted to a channel."""

    channel_id: str
    sender_id: str
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


__all__ = ["User", "Channel", "Message"]
