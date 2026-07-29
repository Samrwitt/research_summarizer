from django.urls import path

from .views import AskDocumentView, DocumentListView, UploadDocumentView

urlpatterns = [
    path("documents/", DocumentListView.as_view(), name="documents"),
    path("documents/upload/", UploadDocumentView.as_view(), name="document-upload"),
    path("documents/<str:ai_document_id>/ask/", AskDocumentView.as_view(), name="document-ask"),
]
