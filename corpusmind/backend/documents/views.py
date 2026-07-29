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


class IntelligenceStatusView(APIView):
    def get(self, request):
        try:
            ai_response = requests.get(f"{settings.AI_SERVICE_URL}/health", timeout=5)
            ai_status = ai_response.json() if ai_response.ok else {"status": "unavailable"}
        except requests.RequestException:
            ai_status = {"status": "unavailable"}

        return Response(
            {
                "backend": "ok",
                "ai_service": ai_status,
                "capabilities": [
                    "document_upload_orchestration",
                    "bert_retrieval",
                    "ocr_ingestion",
                    "table_extraction",
                    "grounded_question_answering",
                    "citation_ranking",
                ],
            }
        )


class UploadDocumentView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request):
        upload = request.FILES.get("file")
        if not upload:
            return Response({"detail": "A file upload is required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            response = requests.post(
                f"{settings.AI_SERVICE_URL}/analyze",
                files={"file": (upload.name, upload.read(), upload.content_type)},
                timeout=240,
            )
        except requests.RequestException as exc:
            return Response(
                {"detail": "AI service is unavailable.", "error": str(exc)},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
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
        question = request.data.get("question", "").strip()
        if not question:
            return Response({"detail": "A question is required."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            response = requests.post(
                f"{settings.AI_SERVICE_URL}/documents/{ai_document_id}/ask",
                json={"question": question, "top_k": request.data.get("top_k", 8)},
                timeout=120,
            )
        except requests.RequestException as exc:
            return Response(
                {"detail": "AI service is unavailable.", "error": str(exc)},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        return Response(response.json(), status=response.status_code)
