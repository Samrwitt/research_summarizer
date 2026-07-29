from django.db import models


class ResearchDocument(models.Model):
    title = models.CharField(max_length=255)
    ai_document_id = models.CharField(max_length=64, unique=True)
    summary = models.TextField(blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return self.title
