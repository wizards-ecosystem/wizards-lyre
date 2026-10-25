"""Creation domain services shared by the server, the worker, and the mock.

Pure Python only: nothing here imports FastAPI, SQLite, project storage,
`acestep`, or `torch`, so the worker process and the Remote GPU host image
can import it as freely as the HTTP server does (SPEC.md sec 16.3).
"""
