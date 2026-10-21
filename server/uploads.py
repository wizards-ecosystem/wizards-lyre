"""Shared WAV/MP3 ingest for `/api` and `/v1` (SPEC.md sec 12 Phase 6).

The body is streamed to a temp file and only published at its final path
once it has been accepted whole, so an oversized or failed upload never
appears at a resolvable path.
"""

from __future__ import annotations

from fastapi import UploadFile

from server import storage


async def receive_audio_upload(project_id: str, file: UploadFile) -> dict[str, str]:
    """Store a drag-dropped WAV/MP3 under `projects/<id>/uploads/`.

    Streamed to disk in bounded chunks -- never `await file.read()` in one
    shot -- so `MAX_UPLOAD_BYTES` is enforced as bytes arrive. An oversized
    upload is aborted (and its partial file deleted) mid-stream instead of
    only after the entire body has already been buffered in memory. The
    final path is published with an atomic rename and is exactly what a
    job's `upload_path` already knows how to resolve (SPEC.md sec 8.1).
    """
    with storage.project_lifecycle_lock(project_id):
        tmp_path, dest_path = storage.open_upload_destination(project_id, file.filename or "")
        total_bytes = 0
        try:
            with open(tmp_path, "wb") as out:
                while chunk := await file.read(storage.UPLOAD_CHUNK_BYTES):
                    total_bytes += len(chunk)
                    if total_bytes > storage.MAX_UPLOAD_BYTES:
                        raise ValueError(
                            f"upload too large: exceeds {storage.MAX_UPLOAD_BYTES} bytes"
                        )
                    out.write(chunk)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
        finally:
            await file.close()
        upload_path = storage.finalize_upload(tmp_path, dest_path)
    return {"upload_path": upload_path}
