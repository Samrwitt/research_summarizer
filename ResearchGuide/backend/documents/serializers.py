from rest_framework import serializers

from .models import ResearchDocument


class ResearchDocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = ResearchDocument
        fields = ["id", "title", "ai_document_id", "summary", "metadata", "created_at"]
