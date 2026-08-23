# Chat Plugin (Backend)

AI-powered chat feature using token-gated LLM access.

## Purpose

Provides a conversational AI endpoint gated by the token system. Users spend tokens per chat message. Supports configurable LLM backends.

## Configuration (`plugins/config.json`)

```json
{
  "chat": {
    "provider": "openai",
    "api_key": "sk-...",
    "model": "gpt-4o",
    "tokens_per_message": 10,
    "max_context_messages": 20
  }
}
```

## API Routes

All routes are prefixed with the plugin's `get_url_prefix()`.

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/send` | Bearer | Send a chat message (deducts tokens) |
| GET | `/config` | Bearer | Get chat configuration for frontend |
| POST | `/estimate` | Bearer | Estimate token cost before sending |

## Events

Consumes: `SubscriptionActivatedEvent` to reset per-period chat allowances.

## Database

Owns tables for conversation history (see migrations in `plugins/chat/migrations/`).

## Frontend Bundle

- User: `vbwd-fe-user/plugins/chat/` (if present)

## Testing

```bash
docker compose run --rm test python -m pytest plugins/chat/tests/ -v
```

---

## Related

| | Repository |
|-|------------|
| 👤 Frontend (user) | [vbwd-fe-user-plugin-chat](https://github.com/VBWD-platform/vbwd-fe-user-plugin-chat) |

**Core:** [vbwd-backend](https://github.com/VBWD-platform/vbwd-backend)

## Documentation

Full platform documentation lives at **[vbwd.cc/docs](https://vbwd.cc/docs)**.

- [Plugin system](https://vbwd.cc/docs-plugin-system) — how backend plugins are registered, enabled, and configured
- [Chat / meinchat](https://vbwd.cc/docs-core-meinchat) — documentation for this plugin's domain
- [Architecture](https://vbwd.cc/docs-architecture) — platform layering and the core-agnosticism rule
- [Getting started](https://vbwd.cc/docs-getting-started) — install a VBWD instance and enable plugins
