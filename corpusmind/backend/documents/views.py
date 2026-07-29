from django.conf import settings
import requests
from rest_framework import status
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import ResearchDocument
from .serializers import ResearchDocumentSerializer


class DocumentListView(APIView):
    def get(self, request):
        documents = ResearchDocument.objects.order_by("-created_at")[:50]
        return Response(ResearchDocumentSerializer(documents, many=True).data)


class UploadDocumentView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request):
        upload = request.FILES.get("file")
        if not upload:
            return Response({"detail": "A file upload is required."}, status=status.HTTP_400_BAD_REQUEST)

        response = requests.post(
            f"{settings.AI_SERVICE_URL}/analyze",
            files={"file": (upload.name, upload.read(), upload.content_type)},
            timeout=240,
        )
        if response.status_code >= 400:
            return Response(response.json(), status=response.status_code)

        payload = response.json()
        document, _ = ResearchDocument.objects.update_or_create(
            ai_document_id=payload["document_id"],
            defaults={
                "title": payload["title"],
                "summary": payload["summary"],
                "metadata": {
                    "ai": payload.get("rag", {}),
                    "source": payload.get("metadata", {}),
                    "insights": payload.get("insights", {}),
                    "bullets": payload.get("bullets", []),
                },
            },
        )
        return Response(
            {"document": ResearchDocumentSerializer(document).data, "analysis": payload},
            status=status.HTTP_201_CREATED,
        )


class AskDocumentView(APIView):
    def post(self, request, ai_document_id: str):
        response = requests.post(
            f"{settings.AI_SERVICE_URL}/documents/{ai_document_id}/ask",
            json={"question": request.data.get("question", ""), "top_k": request.data.get("top_k", 8)},
            timeout=120,
        )
        return Response(response.json(), status=response.status_code)
