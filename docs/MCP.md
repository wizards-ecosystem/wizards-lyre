# MCP

Lyre speaks [MCP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports)
revision 2026-07-28 so Claude Desktop, Cursor, and any other MCP host can use
it as a music tool. The server is a thin client of the [HTTP API](API.md). It
does not open the jobs database and it does not load ACE-Step. Start
`./scripts/lyre server` (and the worker) first. The contract is
[SPEC.md](../SPEC.md) section 15.6.

There is no MCP OAuth. When the HTTP server requires a key, stdio reads
`LYRE_API_KEY` and `POST /mcp` uses the same `Authorization: Bearer` header
as `/v1`. See [Configuration](CONFIGURATION.md).

## Tools

| Tool | What it does |
|---|---|
| `create_song` | `POST /v1/songs`. Sends `Idempotency-Key` (your `idempotency_key`, or a UUID). |
| `transform_take` | Queues `cover`, `repaint`, `extract`, `lego`, or `complete` on an existing project. Sends `Idempotency-Key`. |
| `get_job` | `GET /v1/jobs/{id}`. |
| `cancel_job` | `POST /v1/jobs/{id}/cancel`. A running render stops after the current call. |
| `list_projects` | Projects, newest first. |
| `get_project` | One project, its plan, and its takes. |
| `update_plan` | Replaces `plan.json`. |
| `list_takes` | Takes for one project. |
| `get_artifact` | URL and metadata for `mix.wav`, `mix.mp3`, and `lyrics.lrc`. Does not download the audio. |

`delete_project`, `train_lora`, file upload, and Remote GPU session start are
not tools. Those stay on the HTTP API or on the loopback studio.

## Resources

- `lyre://project/{id}` is the project JSON.
- `lyre://take/{project_id}/{take_id}` is the take JSON.

Audio bytes stay at the artifact URL. A finished generation also includes a
`resource_link` to the take. It does not include an audio content block.

## Tasks and waiting

`create_song` and `transform_take` take minutes.

- If the client advertises `io.modelcontextprotocol/tasks`, omit
  `wait_timeout_sec`. The tool returns a task whose `taskId` is the Lyre job
  id (`status: "working"`, `pollIntervalMs: 2000`). Poll `tasks/get`.
  `tasks/cancel` cancels the job and returns immediately. On Streamable HTTP,
  `tasks/get`, `tasks/update`, and `tasks/cancel` set `Mcp-Name` to that task
  id.
- If the client does not advertise tasks, pass `wait_timeout_sec` (for
  example `600`). The tool enqueues and polls until the job finishes or the
  timeout elapses, then returns the job. A timeout is the current job plus a
  note, not an error. A job status of `error` is a tool error. The value must
  be a finite number from 0 through 86400.
- Omitting `wait_timeout_sec` when the client did not advertise tasks returns
  JSON-RPC `-32021` and does not enqueue.

`get_artifact` returns, for each of the three files, a relative `url`, an
`absolute_url`, `media_type`, `available`, `content_length`, and
`http_status`. `absolute_url` uses `LYRE_URL` for stdio. `POST /mcp` uses
`http://{bind}:{port}`.

## stdio

```sh
./scripts/lyre mcp
```

The process speaks newline-delimited JSON-RPC on stdin and stdout. It calls
`LYRE_URL` (default `http://127.0.0.1:8421`).

### Claude Desktop

Add this to the Claude Desktop config (`claude_desktop_config.json`). Replace
the path with the absolute path to `scripts/lyre` in your checkout.

```json
{
  "mcpServers": {
    "lyre": {
      "command": "/absolute/path/to/wizards-lyre/scripts/lyre",
      "args": ["mcp"],
      "env": {
        "LYRE_URL": "http://127.0.0.1:8421"
      }
    }
  }
}
```

When the HTTP server requires auth, add the secret to `env`:

```json
{
  "mcpServers": {
    "lyre": {
      "command": "/absolute/path/to/wizards-lyre/scripts/lyre",
      "args": ["mcp"],
      "env": {
        "LYRE_URL": "http://127.0.0.1:8421",
        "LYRE_API_KEY": "lyre_..."
      }
    }
  }
}
```

Creating a song needs the `generate` scope. Listing projects needs `read`.
The default key from `./scripts/lyre api-key create` has
`read,generate,write`.

Claude Desktop does not advertise the tasks extension. The server instructions
tell the model to pass `wait_timeout_sec`.

### Cursor

The same stdio block goes in `.cursor/mcp.json` or in Cursor Settings, MCP.

```json
{
  "mcpServers": {
    "lyre": {
      "command": "/absolute/path/to/wizards-lyre/scripts/lyre",
      "args": ["mcp"],
      "env": {
        "LYRE_URL": "http://127.0.0.1:8421"
      }
    }
  }
}
```

Add `LYRE_API_KEY` under `env` when auth is on, the same way as Claude Desktop.

## Streamable HTTP

The HTTP server accepts one stateless `POST /mcp` on the same port as `/v1`
(`http://127.0.0.1:8421/mcp` by default). Every JSON-RPC message is its own
POST. There is no session, no `Mcp-Session-Id`, no GET stream, and no legacy
HTTP+SSE transport. Responses are JSON.

When auth is on, send `Authorization: Bearer <key>`. `POST /mcp` forwards
that token to `/v1`. It does not read `LYRE_API_KEY`.

A missing `Origin` is allowed, which is what native MCP clients send. A
present `Origin` must be loopback (`http` or `https`, host `localhost`,
`127.0.0.1`, or `[::1]`, any port) or listed in `LYRE_ALLOWED_ORIGINS`. Any
other origin is `403` in the `/v1` error shape (`error.code` `forbidden`),
so a web page on another site cannot `POST` to the loopback server and queue
a job while auth is off.

Cursor can point at the HTTP endpoint instead of launching stdio:

```json
{
  "mcpServers": {
    "lyre": {
      "url": "http://127.0.0.1:8421/mcp"
    }
  }
}
```

When auth is on, add the header:

```json
{
  "mcpServers": {
    "lyre": {
      "url": "http://127.0.0.1:8421/mcp",
      "headers": {
        "Authorization": "Bearer lyre_..."
      }
    }
  }
}
```
