# RelayChat

RelayChat is a fully open-source instant messaging engine built entirely in
Python. It focuses on clarity, reproducibility, and approachability, making it a
useful starting point for experiments, hackathons, or as a reference when
teaching the fundamentals of messaging systems.

## Highlights

* **Modern feature set** – register users, create private or public channels, and
  keep presence information in sync.
* **Rich message handling** – track unread counts, search through conversation
  history, and sync read receipts across participants.
* **Batteries included CLI** – the `relaychat` command offers a tiny REPL-like
  experience for demos or smoke testing a deployment.

## Installation

```bash
pip install .
```

## Library usage

```python
from instamsg import InstantMessagingApp

app = InstantMessagingApp()
alice = app.register_user("alice")
bob = app.register_user("bob")
channel = app.create_channel("general", [alice.id, bob.id])
app.send_message(channel.id, alice.id, "Hello from RelayChat!")
```

## Command-line usage

```bash
relaychat --state state.json register alice
relaychat --state state.json register bob --status away
relaychat --state state.json channel create general alice bob
relaychat --state state.json send <channel-id> alice "Hello from the CLI"
relaychat --state state.json history <channel-id>
```

The CLI stores its state inside the JSON file referenced by `--state`, making it
trivial to script or inspect using other tools.
