# ADR 0001: An agent API for Lyre

**Status:** Accepted as a spec amendment on 2026-10-20. Partially implemented (typed `/v1` reads and writes, job object, cooperative cancel, opt-in API keys, one-shot `POST /v1/songs`, MCP on stdio and stateless Streamable HTTP). SSE, webhooks, signed artifact tokens, and `/api` deprecation headers are not built yet.
The normative contract is [SPEC.md](../../SPEC.md) section 15. This note
records why that section looks the way it does, which alternatives were
rejected, and which defaults a maintainer might still overturn. When this
note and SPEC.md disagree, SPEC.md wins.

## Context

Lyre is a local ACE-Step 1.5 studio. The human workflow (plan, takes,
waveform, region edits) is the product, and phases 1–6 of SPEC.md are built.
The HTTP surface in front of that studio is about 23 `/api/*` routes
([API.md](../API.md)): FastAPI, a SQLite queue (`queued` → `running` →
`done` | `error`), poll for status, bind `127.0.0.1`, no auth. The worker is
a separate process and the only place CUDA is loaded. An opt-in Remote GPU
lane can render the same adapter on a GPU the operator rents or owns; a
Runpod pod starts only when the human asks.

That shape is a good studio and a poor tool for an AI agent:

- A song from a prompt takes three calls (create project, rely on the query
  field, enqueue, then poll), and nothing in the contract says the result is
  one operation.
- Progress is "call GET again." There is no event stream, no webhook, and no
  cancel.
- Audio URLs are project-scoped routes that content-negotiate. Agents want a
  path that still names the same bytes next week.
- Responses are untyped dicts. The OpenAPI document exists because FastAPI
  generates one, and it does not pin response schemas.
- The spec forbids auth and any bind beyond loopback, so an agent on another
  machine cannot be let in on purpose. It also has to be kept out by
  accident, which the current text does by forbidding the door.
- There is no MCP server, so clients that speak tools rather than REST have
  nothing to connect to.

The maintainer's bar is that Lyre should be a stronger music tool than a
hosted one-prompt service, and that any AI should be able to use it. The
studio is how Lyre is stronger: lineage, seeds, region repaint, stems, style
packs, weights on a machine the operator controls. The API has to expose that
studio, not replace it with a prompt box.

## Decision

Add section 15 to SPEC.md and implement it as phases 7–13, one pull request
each. The decisions that fix the shape:

1. **Loopback with no auth stays the default.** A network bind is opt-in and
   the process refuses to start unless an API key exists. `LYRE_REQUIRE_AUTH=1`
   covers the case where a tunnel or reverse proxy makes remote traffic look
   like loopback. The SPA and `/api` are served only on the auth-off loopback
   server, so exposing the API does not expose the studio UI.
2. **Keys are operator-minted bearer secrets**, hashed at rest, with scopes
   `read`, `generate`, and `write`. They are not accounts. The MCP OAuth
   flow in the 2026-07-28 spec is optional, and this amendment does not
   implement it. stdio takes `LYRE_API_KEY` from the environment, which is
   what that spec tells stdio servers to do.
3. **`/v1` is a typed, versioned surface** over the same queue and the same
   projects. `/api` freezes at phase 7 (bugfixes only), gains deprecation
   headers when the SPA moves in phase 13, and is not removed by this
   amendment.
4. **`POST /v1/songs` is one-shot simple or custom generation.** It writes a
   normal project. The guided loop remains available as ordinary job posts.
   Caller-supplied caption or lyrics default to `caption_rewrite: false`.
5. **Progress is SSE for clients that can hold a connection, and a webhook
   for clients that cannot.** MCP clients use the tasks extension and poll
   `tasks/get`. Webhooks fire on terminal status only.
6. **Cancel is cooperative.** ACE-Step cannot interrupt a running diffusion
   call. The worker discards the result afterward. The remote host already
   behaves this way.
7. **Artifact paths are stable and relative.** Headerless players get a
   10-minute MAC-signed query token for one file. API keys do not go in the
   query string.
8. **MCP speaks revision 2026-07-28** on stdio and on stateless Streamable
   HTTP (`POST /mcp`). Generation tools return `io.modelcontextprotocol/tasks`
   handles whose `taskId` is the Lyre job id, or JSON-RPC `-32021` if the
   client cannot handle tasks and did not pass `wait_timeout_sec`. Passing
   `wait_timeout_sec` makes the server poll and return a complete result,
   which is how a client that does not advertise tasks still creates a song.
   Mix audio is a URL, not a base64 content block. The adapter is a thin
   client of `/v1` (`LYRE_URL` and `LYRE_API_KEY` on stdio); it does not
   open the jobs database.
9. **The worker and the Remote GPU lane are unchanged in role.** New columns
   are `stage` and `cancel_requested`. A queued agent job never calls the
   Runpod provisioner. Session start stays a human action on the loopback
   studio. Style packs stay local-only.

## Alternatives considered

**Keep `/api` and document a client recipe.** Rejected. Agents would still
be polling untyped dicts, and the spec would still forbid letting one in
from another machine. A recipe is not a contract.

**Replace `/api` in place.** Rejected. The studio, the live-check script,
and the route test all speak `/api` today. A parallel `/v1` lets those keep
working until phase 13 moves them.

**Make one-shot the whole product.** Rejected. A single prompt cannot
repaint a region or walk a take lineage. Those are the reasons to use Lyre.
The one-field page stays out of the UI on purpose.

**Block `tools/call` until the song is done.** Rejected. Renders outlast
typical client timeouts, and the tasks extension exists for this. The
2026-07-28 tasks spec requires `-32021` when the server cannot proceed
without the extension, which matches a render that must not block.

**stdio only, or Streamable HTTP only.** Both are required. stdio is how a
local IDE launches a tool without opening a port. Streamable HTTP is how an
agent reaches a server the operator deliberately bound to a network. The
removed HTTP+SSE transport (protocol 2024-11-05) and the pre-2026-07-28
sessionful Streamable HTTP are not implemented; new clients speak the
stateless revision.

**MCP OAuth / CIMD as the only auth.** Rejected for this amendment. Lyre is
one operator minting keys for their own agents. OAuth's authorization
server, protected-resource metadata, and client-id metadata documents are
a second identity system. Bearer keys cover HTTP and the stdio environment
rule. Revisit only if a client cannot send a bearer token.

**Serve the studio on the network bind behind a cookie session.** Rejected
for now. It pulls account-like session machinery into a tool that has one
operator. An SSH tunnel to loopback already reaches the studio.

**Inline WAV in MCP `audio` content.** Rejected. A two-minute mix does not
belong in a stdio JSON-RPC message. The artifact URL is the byte pipe.

**Auto-start a Remote GPU when an agent enqueues.** Rejected. It spends
money because a job exists, which section 3.1 forbids. The human starts the
pod. The agent uses it if it is already up.

**Hard-cancel by killing the worker process.** Rejected. That drops the
GPU lease and fails every other queued job. Cooperative cancel matches what
the remote host can actually do.

## Consequences

- SPEC.md section 2 no longer forbids auth outright. It forbids accounts,
  multi-tenant hosting, and a cloud deployment of Lyre. SECURITY.md,
  the README, and the contributor docs describe the default as today's
  behavior and point at section 15 for the opt-in.
- `docs/API.md` stays the list of routes the process serves. New `/v1`
  rows land in the same pull request that mounts them, because
  `tests/test_docs_match_code.py` fails when the table and the app differ.
- `docs/CONFIGURATION.md` does not list `LYRE_BIND` or the key settings
  until the phase 11 code reads them. The same test fails on a documented
  variable nothing reads.
- The default bind constant in source remains `127.0.0.1`. Phase 11 must
  not introduce a public-bind literal; `tests/test_spec_lock.py` forbids
  `0.0.0.0` in source.
- Phases 9 and 13 need a real GPU via `./scripts/lyre live-check`. Every
  other phase, including MCP, is covered by the mocked worker.

## Open questions

The defaults in SPEC.md section 15 are normative. They are listed here so a
maintainer can overturn one before its phase starts, in a spec PR, without
re-opening the rest.

1. **Remote GPU session start from an agent.** Left off `/v1` and off the
   MCP tool list, so a queued song cannot rent a pod. Confirm that a
   scoped, explicit rent call should stay a human action in the studio.
2. **MCP OAuth.** Bearer keys only. Confirm that the agents to support
   (local stdio, and HTTP clients that can send `Authorization`) do not
   need the 2026-07-28 OAuth / client-id metadata flow.
3. **Inlined audio.** Tool results carry the job JSON and a `resource_link`,
   not an `audio` content block. Confirm no target client requires the
   base64 block.
4. **`train_lora` and uploads as MCP tools.** Left on HTTP only. Training
   is about an hour and local-only; upload is a binary body. Confirm they
   stay off the default tool list.
5. **Webhook granularity.** Terminal transitions only. Confirm a webhook
   per `stage` is unnecessary.
6. **Signed query tokens.** 10-minute MAC, one file, so `<audio>` works
   when auth is on. Confirm a token in the query string is acceptable,
   knowing it can land in access logs.
7. **`/api` removal.** Headers in phase 13; no removal date. Confirm this
   amendment should not name a sunset release.
8. **Remote studio.** Not served on a network bind. Confirm an SSH tunnel
   to loopback is enough, and that a cookie session for the SPA is out of
   scope.
