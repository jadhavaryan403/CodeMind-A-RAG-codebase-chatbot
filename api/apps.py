# api/apps.py
from django.apps import AppConfig
class ApiConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "api"

# rag_core/apps.py
from django.apps import AppConfig
class RagCoreConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "rag_core"
