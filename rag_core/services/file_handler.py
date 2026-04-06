"""
file_handler.py

Secure file validation and storage.
Supports both single file uploads and folder uploads.

Security guarantees:
- Extension whitelist (.py only)
- Size limit enforcement
- UUID-based stored filenames (never raw user input)
- Relative paths are sanitised — no path traversal possible
- All paths derived from integer PKs only
"""

import uuid
import os
from pathlib import Path, PurePosixPath
from django.conf import settings
from django.core.exceptions import ValidationError


# ── Validation ────────────────────────────────────────────────────────────────

def validate_upload(file) -> None:
    """
    Raise ValidationError if the upload fails security checks.
    Call this BEFORE touching the filesystem.
    """
    # 1. Extension whitelist
    suffix = Path(file.name).suffix.lower()
    if suffix not in settings.ALLOWED_UPLOAD_EXTENSIONS:
        raise ValidationError(
            f"Invalid file type '{suffix}'. "
            f"Only {settings.ALLOWED_UPLOAD_EXTENSIONS} are allowed."
        )

    # 2. Size limit
    if file.size > settings.MAX_UPLOAD_SIZE_BYTES:
        max_mb = settings.MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
        raise ValidationError(
            f"File '{file.name}' exceeds the {max_mb} MB limit."
        )


def sanitise_relative_path(relative_path: str) -> str:
    """
    Sanitise a browser-supplied relative path (webkitRelativePath).

    Prevents path traversal attacks like:
        ../../etc/passwd
        ../secret.py
        /absolute/path.py

    Returns a safe relative path string like:
        myproject/utils/helpers.py
        views.py

    Rules applied:
        - Convert to forward slashes
        - Remove any leading slashes
        - Resolve and strip all '..' components
        - Keep only the sanitised path segments
    """
    if not relative_path:
        return ""

    # Normalise separators
    relative_path = relative_path.replace("\\", "/")

    # Use PurePosixPath to safely handle the path
    try:
        parts = PurePosixPath(relative_path).parts
    except Exception:
        return ""

    # Remove dangerous components
    safe_parts = []
    for part in parts:
        if part in (".", "..", "/", ""):
            continue
        # Strip any remaining leading dots or slashes from individual parts
        clean = part.lstrip("./")
        if clean:
            safe_parts.append(clean)

    return "/".join(safe_parts)


# ── Save ──────────────────────────────────────────────────────────────────────

def save_uploaded_file(
    file,
    user_id: int,
    project_id: int,
    relative_path: str = "",
) -> tuple[str, int]:
    """
    Safely persist `file` to disk.

    Args:
        file:          The uploaded file object from request.FILES
        user_id:       Django user.id (integer — never user-supplied string)
        project_id:    Project.pk (integer — never user-supplied string)
        relative_path: Optional browser-supplied webkitRelativePath
                       e.g. "myproject/utils/helpers.py"
                       This is sanitised before use — never used raw.

    Returns:
        (stored_path, size_bytes)
        stored_path is relative to settings.UPLOAD_ROOT

    Directory structure created:
        Single file:
            uploads/user_1/project_3/abc123.py

        Folder upload (relative_path="myapp/utils/helpers.py"):
            uploads/user_1/project_3/myapp/utils/<uuid>.py

    Security:
        - relative_path is always sanitised via sanitise_relative_path()
        - Filename is always a UUID — original name never touches the filesystem
        - Directory path uses integer IDs + sanitised folder structure only
    """
    validate_upload(file)

    # Base directory using integer IDs only
    base_dir = (
        settings.UPLOAD_ROOT
        / f"user_{user_id}"
        / f"project_{project_id}"
    )

    # If a relative path was provided, preserve the folder structure
    # but strip the filename from it (we use UUID filename instead)
    if relative_path:
        safe_path   = sanitise_relative_path(relative_path)
        # Get only the directory part — discard the original filename
        folder_part = "/".join(safe_path.split("/")[:-1])
        if folder_part:
            dest_dir = base_dir / folder_part
        else:
            dest_dir = base_dir
    else:
        dest_dir = base_dir

    dest_dir.mkdir(parents=True, exist_ok=True)

    # UUID filename — original filename never used on filesystem
    safe_name = f"{uuid.uuid4().hex}.py"
    dest_path = dest_dir / safe_name

    content = file.read()
    dest_path.write_bytes(content)

    # Return path relative to UPLOAD_ROOT (stored in DB)
    relative = str(dest_path.relative_to(settings.UPLOAD_ROOT))
    return relative, len(content)