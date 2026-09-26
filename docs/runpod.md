# Running the Remote GPU on Runpod

[The Remote GPU lane](remote-gpu.md) connects Lyre to an ACE-Step host you
operate. This guide covers one way to get the hardware: renting it from Runpod
by the second, with Lyre starting and stopping the pod for you.

Nothing here is required. The host image is provider-neutral, and you can
always run it yourself and paste its address instead. You are renting the
machine, so the account, the spend, and the model licences are yours.

## Set up once

1. Create an API key at <https://console.runpod.io/user/settings> with
   permission to create and terminate pods.
2. Put the settings in `.env` in the checkout (gitignored; `scripts/lyre`
   loads it), then start the server:

   ```bash
   umask 077
   cat >> .env <<'EOF'
   LYRE_REMOTE_GPU_PROVISIONER=runpod
   LYRE_RUNPOD_API_KEY=rpa_...
   EOF
   ./scripts/lyre server
   ```

3. Start the worker in remote mode, in another terminal:

   ```bash
   LYRE_WORKER=remote ./scripts/lyre worker
   ```

The key is read only by the server and never leaves your machine. Every other
setting has a default; see [CONFIGURATION.md](CONFIGURATION.md#remote-gpu-variables).

## Start a GPU

The **Remote GPU** panel shows the GPU type and its advertised hourly rate
before you press anything. **Start GPU** creates one pod, and while it runs the
panel shows elapsed time and estimated spend. Lyre connects to the pod's
address with a secret made for that session.

**Nothing provisions on its own.** There is no timer, no start-on-launch, and no
"rent a GPU because a job was queued". A second Start while one is up is
refused rather than billed twice.

The pod runs the published host image that matches your checkout:
`ghcr.io/wizards-ecosystem/wizards-lyre-remote-gpu:<build id>`. If you changed
`worker/remote_host` or `worker/acestep_worker`, no published image matches.
Lyre refuses to start a pod that could never pull its image, and tells you to
build your own (see below).

### What "starting" means

In order, while you wait:

1. **pod provisioning**: Runpod is finding a machine.
2. **pulling the host image**: the first pull of a ~13 GB image on that
   machine. This is the long wait.
3. **Remote GPU starting**: the host is downloading the core weights (~10 GB)
   and loading ACE-Step.
4. **ready**: the pod is up and ACE-Step answers.

The first job on a profile other than `iterate` also downloads that profile's
checkpoint before it renders.

## Stopping it is your job

Press **Stop GPU**, or quit the server cleanly and it terminates the pod for
you (`LYRE_REMOTE_GPU_STOP_ON_EXIT`, on by default). Stop always
*terminates*: a merely stopped pod gives up its GPU with no promise of getting
it back, and keeps billing its disk.

That is the whole safety net. There is no idle timer and no self-destruct. A
crash, a killed process, or a closed laptop runs no shutdown code, and the pod
keeps billing until something stops it. If the server starts and finds a pod
from an earlier run, it logs a warning with the elapsed time and spend, and the
panel shows it with a Stop button. Only a pod this install recorded is ever
treated as its own. Anything else in your Runpod account is left alone.

Runpod's own account spend limit is a sensible backstop. Set one in the
console.

## Choosing a GPU

Lyre is tuned for a 16 GB card, and every profile, XL included, fits in 24 GB
without offload. Prices and stock from Runpod's catalog on **2026-09-26**,
Secure Cloud. Re-read them before you commit; they move:

| GPU | VRAM | $/hr |
|---|---|---|
| **RTX 4090** (default) | 24 GB | **0.74** |
| RTX A5000 | 24 GB | 0.27 |
| L4 | 24 GB | 0.49 |
| RTX 5090 | 32 GB | 0.99 |
| L40S | 48 GB | 1.09 |

The A5000 is the cheapest 24 GB card, but it had no Secure Cloud stock on
any read that day, so it is not the default; set
`LYRE_RUNPOD_GPU_TYPE="NVIDIA RTX A5000"` to try it. Stock for any one type
comes and goes. If a start fails for lack of capacity,
set `LYRE_RUNPOD_GPU_TYPE` to another type id, or
`LYRE_RUNPOD_CLOUD=COMMUNITY` for cheaper, less isolated hosts. Pods are always
placed on hosts that support CUDA 12.8, which the pinned PyTorch needs.

Render times on these cards have not been measured yet.

## Keeping the weights

By default the weights live on the pod's container disk and are lost when it
is terminated, so each new pod downloads them again (a few minutes for the core
set). Set `LYRE_RUNPOD_NETWORK_VOLUME_ID` to keep them on a network volume
instead. That costs storage every month and pins every pod to the volume's data
center, which narrows GPU availability. It buys waiting less, not spending
less.

## Building your own image

Needed when you changed the host code, or want your own registry:

```bash
./scripts/lyre remote-image
docker tag wizards-lyre-remote-gpu:<build id> <registry>/wizards-lyre-remote-gpu:<build id>
docker push <registry>/wizards-lyre-remote-gpu:<build id>
export LYRE_RUNPOD_IMAGE=<registry>/wizards-lyre-remote-gpu:<build id>
```

For a private registry, add its credentials in the Runpod console under
Settings -> Container Registry. Never tag the image `latest`: hosts cache
images per machine, so a mutable tag silently serves stale code.

To smoke-test an image without spending GPU time on models, run it with
`LYRE_REMOTE_BACKEND=mock`. It answers the full protocol and renders silence.

## Troubleshooting

**Stuck on "pulling the host image".** A first pull on a new machine takes
minutes. If it never finishes, check the pod's logs in the console. A wrong or
private `LYRE_RUNPOD_IMAGE` never pulls, and still bills until you stop it.

**Your code changes did nothing.** You are running an older image. The panel
flags a host whose build id differs from your checkout.

**"No published host image matches this checkout".** See
[Building your own image](#building-your-own-image).

**"Runpod rejected the API key".** Check `LYRE_RUNPOD_API_KEY`. A read-only key
can show prices but cannot create or terminate pods.
