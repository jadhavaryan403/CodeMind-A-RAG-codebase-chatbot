"""
serializers.py — DRF serializers

ProjectSerializer:  Create / list projects.
QuerySerializer:    Validate incoming Q&A requests.
"""

from rest_framework import serializers
from rag_core.models import Project, UploadedFile, Project, UploadedFile, Conversation, Message


class ProjectSerializer(serializers.ModelSerializer):
    class Meta:
        model  = Project
        fields = ["id", "name", "description", "status", "error_message",
                  "created_at", "updated_at", "github_url"]
        read_only_fields = ["id", "status", "error_message", "created_at", "updated_at"]


class UploadedFileSerializer(serializers.ModelSerializer):
    class Meta:
        model  = UploadedFile
        fields = ["id", "original_filename", "file_size_bytes", "uploaded_at"]


class QuerySerializer(serializers.Serializer):
    query = serializers.CharField(min_length=3, max_length=1000)



class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model  = Message
        fields = ["id", "role", "content", "cited_chunks", "created_at"]
        read_only_fields = fields


class ConversationSerializer(serializers.ModelSerializer):
    messages      = MessageSerializer(many=True, read_only=True)
    message_count = serializers.SerializerMethodField()

    class Meta:
        model  = Conversation
        fields = ["id", "title", "message_count", "messages", "created_at", "updated_at"]
        read_only_fields = ["id", "message_count", "messages", "created_at", "updated_at"]

    def get_message_count(self, obj):
        return obj.messages.count()


class ConversationListSerializer(serializers.ModelSerializer):
    """Lightweight serializer for the sidebar list — no messages included."""
    message_count = serializers.SerializerMethodField()
    last_message  = serializers.SerializerMethodField()

    class Meta:
        model  = Conversation
        fields = ["id", "title", "message_count", "last_message", "created_at", "updated_at"]

    def get_message_count(self, obj):
        return obj.messages.count()

    def get_last_message(self, obj):
        msg = obj.messages.last()
        return msg.content[:80] if msg else None