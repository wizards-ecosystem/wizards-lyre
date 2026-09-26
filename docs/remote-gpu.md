# Remote GPU

The Wizard's Lyre runs ACE-Step 1.5 on your own GPU, and that stays the
default. The Remote GPU lane is an opt-in alternative for when you do not have
a suitable card: the same ACE-Step 1.5, driven by the same adapter code, running
on a GPU you rent or run elsewhere. It is not a hosted service. This repository
provides no GPU access, and nothing here contacts another music engine.

You are the operator of that GPU. The account, the spend, the network exposure,
and the model licences are yours.

## How it fits

```
Browser -> server (127.0.0.1) -> SQLite queue -> worker (LYRE_WORKER=remote)
                                                     |
                                                     |  HTTPS + shared secret
                                                     v
                                     Remote GPU host (worker.remote_host)
                                                     |
                                                     v
                                     worker.acestep_worker -> ACE-Step 1.5
```

Nothing about projects, takes, or the queue changes. The local worker still
claims jobs, holds the one-GPU lease, and writes every take into `projects/`.
It just hands the render to the host instead of loading ACE-Step itself. Takes
never live on the remote machine: the host keeps a result only until Lyre has
downloaded it.

The host is the package `worker/remote_host`, shipped as a container image
(`docker/remote-gpu/Dockerfile`). It holds no secrets and no weights. It fetches
each profile's weights the first time a job needs them.

## What works remotely

| Works | Not yet |
|---|---|
| Generate, cover, repaint | Style packs: training a LoRA, or generating with one |
| Extract, lego, complete (the base-model swap) | |
| Every DiT profile, including XL `quality` on a 24 GB card | |

A job that asks for a style pack fails with a message saying so. Switch back
to the local worker for those.

## Getting a host

**Rent one on Runpod.** Lyre can start and stop the pod for you. See
[runpod.md](runpod.md).

**Run it on a machine you own**, for example a desktop with a GPU you reach from
a laptop. On the GPU machine, in a Lyre checkout with ACE-Step installed
(`./scripts/lyre install`):

```bash
export LYRE_REMOTE_SECRET="$(openssl rand -hex 32)"
echo "$LYRE_REMOTE_SECRET"          # you will enter this in Lyre
./scripts/lyre remote-host
```

It binds `127.0.0.1:8000` and reuses that checkout's downloaded weights. Reach it
through an SSH tunnel from the machine running Lyre, and connect to
`http://127.0.0.1:8000`, the only plain-HTTP address Lyre accepts:

```bash
ssh -N -L 8000:127.0.0.1:8000 you@gpu-machine
```

**Run the container anywhere else** that gives it an NVIDIA GPU and an HTTPS
address. Build it with `./scripts/lyre remote-image`, and pass
`LYRE_REMOTE_SECRET` at deploy time. See [CONFIGURATION.md](CONFIGURATION.md#remote-gpu-variables)
for the host's variables.

## Connecting Lyre

1. Start the worker in remote mode instead of the local one:

   ```bash
   LYRE_WORKER=remote ./scripts/lyre worker
   ```

2. Open the **Remote GPU** panel in the studio header. Enter the host's address
   and secret, or press **Start GPU** if a provisioner is configured.

The worker checks the connection on every heartbeat, so connecting,
disconnecting, or replacing a host needs no restart. Until a host is ready,
`/api/health` says why, and a job fails with that reason rather than hanging.

The same connection can be set through the API: `PUT /api/remote-gpu/connection`
with `{ "base_url": ..., "secret": ... }`. See [API.md](API.md#remote-gpu).

## Security

- **The secret is the lock.** A provider's proxy address is public, and the
  host authenticates every request with the shared secret before reading its
  body. It refuses to start with a missing, short, or placeholder secret.
- **HTTPS only.** Lyre sends the secret only to an `https://` address, or to
  plain HTTP on this machine itself. It never follows a redirect, so the secret
  cannot be replayed to another address.
- **Stored privately.** The connection is saved in `remote_gpu.json` beside the
  jobs database with owner-only permissions. No API response contains the
  secret.
- **Keys stay home.** A provisioner's API key is read from the server's
  environment and never sent to the pod. A Runpod pod gets a secret generated
  for that session alone.
- **Your server is still local.** The Lyre server itself still binds
  `127.0.0.1` with no authentication. The Remote GPU lane opens nothing on your
  machine.

## Stale hosts

The host reports a build id: a fingerprint of `worker/remote_host` and
`worker/acestep_worker`. When it differs from your checkout's, the panel says
so. The host is still usable, but it runs different rendering code than the
source in front of you. Rebuild and redeploy the image to clear it.

## The host protocol

For anyone writing their own client or host. Every request carries
`X-Lyre-Secret`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Build id, readiness, GPU, loaded profile, per-profile capability, queue depth, free disk. |
| `POST` | `/jobs` | Submit `{ client_id, take_id, job, plan, src_audio_b64?, src_audio_suffix? }`. Returns `{ token }` at once. A repeated `client_id` returns the same token. |
| `GET` | `/jobs/{token}` | `status` (`queued`, `running`, `done`, `error`), `dit_loaded`, and on `done` the take's `meta`, `plan_patch`, `lrc_text`, and audio name and size. |
| `GET` | `/jobs/{token}/audio` | The rendered `mix.wav` or `mix.mp3`. |
| `POST` | `/jobs/{token}/ack` | The client has the take; free it. |
| `POST` | `/jobs/{token}/cancel` | Cancel a job that has not started. |

Submit returns immediately and the client polls, because a render outlasts the
~100 s that proxies hold a request open. Reading a result never deletes it, so a
dropped response is recovered by polling again. Only an ack, or an hour
without one, frees it.

## Troubleshooting

**"no Remote GPU connected".** The worker is in remote mode but no host is
connected. Connect one, or run the worker without `LYRE_WORKER=remote`.

**"Remote GPU starting".** The host is up but still downloading weights or
loading ACE-Step. A first start downloads about 10 GB. Watch the host's log.

**"the Remote GPU rejected the shared secret".** The secret you entered does not
match the host's `LYRE_REMOTE_SECRET`.

**"the Remote GPU no longer knows this job".** The host restarted mid-render and
lost its in-memory queue. Run the job again.

**A 524 from the proxy.** Something called the host directly and waited on a
render. Lyre never does; it polls.
