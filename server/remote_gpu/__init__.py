"""The optional Remote GPU lane: ACE-Step 1.5 on hardware you rent or own.

Off by default. With no connection saved, nothing here makes a network call
and Lyre is the local-only app SPEC.md describes. See docs/remote-gpu.md.

- `connection`   -- the one host this install talks to, remembered on disk
- `client`       -- the HTTP protocol to that host (submit, poll, fetch, ack)
- `provisioners` -- what can make a host exist: `manual` (you) or `runpod`

Unlike `server.storage` and `server.jobs`, this package does not re-export its
modules; import the one you need.
"""
