from django.urls import path

from api.views import (
    frontend_view,
    login_view, 
    logout_view,
    register_view,
    ProjectListCreateView, 
    FileUploadView,
    ProjectFilesView,
    ConversationListCreateView,
    ConversationDetailView,
    ConversationStreamView,
    GithubRepoInfoView,
    GithubImportView,
    ReindexView,
    GithubImportFilesView,
    IndexFilesView,
    ProjectStatusView,
)

urlpatterns = [
    path("login/",                                      login_view),
    path("logout/",                                     logout_view),
    path("register/",                                   register_view),
    path("projects/",                                   ProjectListCreateView.as_view()),
    path("projects/<int:project_id>/upload/",           FileUploadView.as_view()),
    path("projects/<int:project_id>/files/",            ProjectFilesView.as_view()),
    path("projects/<int:project_id>/conversations/",    ConversationListCreateView.as_view()),
    path("conversations/<int:conv_id>/",                ConversationDetailView.as_view()),
    path("conversations/<int:conv_id>/stream/",         ConversationStreamView.as_view()),
    path("github/info/",                                GithubRepoInfoView.as_view()),
    path("projects/<int:project_id>/import-github/",    GithubImportView.as_view()),
    path("projects/<int:project_id>/import-github-files/", GithubImportFilesView.as_view()),
    path("projects/<int:project_id>/index-files/",         IndexFilesView.as_view()),
    path("projects/<int:project_id>/reindex/", ReindexView.as_view()),
    path('projects/<int:project_id>/status/', ProjectStatusView.as_view()),
]