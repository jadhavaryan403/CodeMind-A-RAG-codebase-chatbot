"""
Models — rag_core

Project: a named code collection per user.
  → Each has its own FAISS directory: vectorstores/user_<id>/project_<id>/

UploadedFile: a single .py file belonging to a project.
  → Stored under uploads/ with a UUID filename — never the raw user name.
"""

from django.db import models
from django.contrib.auth import get_user_model

User = get_user_model()


class Project(models.Model):
    class Status(models.TextChoices):
        PENDING  = "pending",  "Pending"
        INDEXING = "indexing", "Indexing"
        READY    = "ready",    "Ready"
        FAILED   = "failed",   "Failed"

    owner         = models.ForeignKey(User, on_delete=models.CASCADE, related_name="projects")
    name          = models.CharField(max_length=255)
    description   = models.TextField(blank=True)
    status        = models.CharField(max_length=20, choices=Status.choices, default=Status.PENDING)
    indexing_step = models.CharField(max_length=100, blank=True, default="")
    error_message = models.TextField(blank=True)
    github_url    = models.URLField(blank=True)
    created_at    = models.DateTimeField(auto_now_add=True)
    updated_at    = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        unique_together = [("owner", "name")]

    def __str__(self):
        return f"{self.owner.username}/{self.name} [{self.status}]"

    @property
    def vectorstore_subpath(self) -> str:
        """
        Safe sub-path derived from integer PKs only.
        Full path is resolved in faiss_store.py using settings.VECTORSTORE_ROOT.
        Never constructed from user-supplied strings.
        """
        return f"user_{self.owner_id}/project_{self.pk}"


class UploadedFile(models.Model):
    project           = models.ForeignKey(Project, on_delete=models.CASCADE, related_name="files")
    original_filename = models.CharField(max_length=255)   # display only — never used in paths
    original_path     = models.CharField(max_length=500, default='')
    stored_path       = models.CharField(max_length=512)   # UUID-based, relative to UPLOAD_ROOT
    file_size_bytes   = models.PositiveIntegerField()
    uploaded_at       = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.original_filename} → {self.project}"


class Conversation(models.Model):
    """A named chat session belonging to a user + project."""
    project   = models.ForeignKey(Project, on_delete=models.CASCADE, related_name="conversations")
    title     = models.CharField(max_length=255, default="New Chat")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self):
        return f"{self.title} [{self.project}]"


class Message(models.Model):
    """A single message inside a Conversation."""

    class Role(models.TextChoices):
        USER      = "user",      "User"
        ASSISTANT = "assistant", "Assistant"

    conversation  = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name="messages")
    role          = models.CharField(max_length=20, choices=Role.choices)
    content       = models.TextField()
    cited_chunks  = models.JSONField(default=list, blank=True)
    created_at    = models.DateTimeField(auto_now_add=True)
    input_tokens   = models.PositiveIntegerField(default=0)
    output_tokens  = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return f"[{self.role}] {self.content[:60]}"


class ChunkIndex(models.Model):
    """
    Tracks every chunk that has been indexed into FAISS for a project.
    Used for incremental re-indexing — only re-process changed chunks.
    """
    project     = models.ForeignKey(
        Project, on_delete=models.CASCADE, related_name='chunk_indices'
    )
    symbol      = models.CharField(max_length=255)      # function/class name
    file_path   = models.CharField(max_length=500)      # relative path in repo
    original_path = models.CharField(max_length=500, default='') # original name of file when uploaded
    chunk_type  = models.CharField(max_length=50)       # function / class / method
    code_hash   = models.CharField(max_length=64)       # SHA256 of code_text
    faiss_id    = models.IntegerField()                 # position in FAISS index
    explanation = models.TextField()                    # stored explanation
    start_line  = models.IntegerField(default=0)
    end_line    = models.IntegerField(default=0)
    updated_at  = models.DateTimeField(auto_now=True)
    input_tokens = models.PositiveIntegerField(default=0)
    output_tokens = models.PositiveIntegerField(default=0)

    class Meta:
        # A symbol is unique per file per project
        unique_together = [('project', 'symbol', 'original_path', 'start_line', 'end_line')]
        indexes = [
            models.Index(fields=['project', 'symbol']),
            models.Index(fields=['project', 'original_path']),
        ]

    def __str__(self):
        return f"{self.symbol} [{self.project.name}]"

