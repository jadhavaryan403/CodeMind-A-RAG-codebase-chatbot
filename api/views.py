import logging
import json
from django.http import StreamingHttpResponse
from django.conf import settings
from django.shortcuts import get_object_or_404, render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.contrib.auth import authenticate 
from django.contrib.auth.models import User
from rest_framework.authtoken.models import Token
from rest_framework.permissions import AllowAny
from rest_framework.decorators import api_view, permission_classes

from rag_core.models import Project, UploadedFile, Conversation, Message
from rag_core.services.file_handler import save_uploaded_file
from rag_core.services.ast_chunker import chunk_file
from rag_core.services.explainer import generate_explanations_batch
from rag_core.services.faiss_store import build_and_save_index
from rag_core.services.incremental_indexer import incremental_reindex
from rag_core.services.github_fetcher import fetch_github_repo, GithubFetchError

from rag_core.langchain.nodes import stream_answer


from api.serializers import (
    ProjectSerializer, UploadedFileSerializer, QuerySerializer,
    ConversationSerializer, ConversationListSerializer,
)

from rag_core.services.github_fetcher import (
    fetch_github_repo,
    get_repo_info,
    GithubFetchError,
)
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def _validate_explanations(explanations, chunks):
    """
    Ensure structured explanations are valid.
    Prevents silent failures in FAISS + metadata.
    """
    valid = []
    for chunk, exp in zip(chunks, explanations):
        if not isinstance(exp, dict):
            valid.append({
                "one_line_summary": f"{chunk.symbol_name} logic",
                "detailed_explanation": chunk.code_text[:200],
                "dependencies": []
            })
            continue

        valid.append({
            "one_line_summary": exp.get("one_line_summary", "")[:100],
            "detailed_explanation": exp.get("detailed_explanation", ""),
            "dependencies": list(set(exp.get("dependencies", [])))
        })
    return valid

@api_view(['POST'])
@permission_classes([AllowAny])
def register_view(request):
    '''Register a new user account. Expects username, email, password, password2.'''

    username   = request.data.get('username', '').strip()
    email      = request.data.get('email', '').strip()
    password   = request.data.get('password', '')
    password2  = request.data.get('password2', '')

    # ── Validation ────────────────────────────────────────────────────────────
    errors = {}

    if not username:
        errors['username'] = 'Username is required.'
    elif len(username) < 3:
        errors['username'] = 'Username must be at least 3 characters.'
    elif len(username) > 30:
        errors['username'] = 'Username must be 30 characters or less.'
    elif not username.isalnum() and not all(c.isalnum() or c in ('_', '-') for c in username):
        errors['username'] = 'Username may only contain letters, numbers, _ and -.'
    elif User.objects.filter(username__iexact=username).exists():
        errors['username'] = 'This username is already taken.'

    if email and User.objects.filter(email__iexact=email).exists():
        errors['email'] = 'An account with this email already exists.'

    if not password:
        errors['password'] = 'Password is required.'
    elif len(password) < 8:
        errors['password'] = 'Password must be at least 8 characters.'

    if not password2:
        errors['password2'] = 'Please confirm your password.'
    elif password and password != password2:
        errors['password2'] = 'Passwords do not match.'

    if errors:
        logger.warning("Validation errors occurred", extra={"errors": errors})
        return Response(errors, status=status.HTTP_400_BAD_REQUEST)

    # ── Create user ───────────────────────────────────────────────────────────
    user = User.objects.create_user(
        username=username,
        email=email,
        password=password,
    )

    logger.info("User created", extra={"user_id": user.id})

    token, _ = Token.objects.get_or_create(user=user)

    return Response({
        'token':    token.key,
        'username': user.username,
        'email':    user.email,
        'message':  'Account created successfully.',
    }, status=status.HTTP_201_CREATED)


@api_view(['POST'])
@permission_classes([AllowAny])
def login_view(request):
    '''Log in an existing user. Expects username and password. Returns auth token on success.'''

    username = request.data.get('username', '').strip()
    password = request.data.get('password', '').strip()

    if not username or not password:
        return Response({'error': 'Username and password are required.'},
                        status=status.HTTP_400_BAD_REQUEST)

    user = authenticate(request, username=username, password=password)
    if not user:
        return Response({'error': 'Invalid username or password.'},
                        status=status.HTTP_401_UNAUTHORIZED)

    token, _ = Token.objects.get_or_create(user=user)
    logger.info("User logged in", extra={"user_id": user.id})

    return Response({
        'token': token.key,
        'username': user.username,
        'email': user.email,
    })


@api_view(['POST'])
def logout_view(request):
    '''Log out the current user by deleting their auth token.'''
    Token.objects.filter(user=request.user).delete()
    logger.info("User logged out", extra={"user_id": request.user.id})
    return Response({'message': 'Logged out successfully.'})

# ── Serves frontend.html at / ─────────────────────────────────────────────────
def frontend_view(request):
    return render(request, "frontend.html")


# ── GET/POST /api/projects/ ───────────────────────────────────────────────────
class ProjectListCreateView(APIView):
    '''GET/POST /api/projects/'''

    def get(self, request):
        '''List all projects for the authenticated user.'''
        projects = Project.objects.filter(owner=request.user)
        return Response(ProjectSerializer(projects, many=True).data)

    def post(self, request):
        '''Create a new project for the authenticated user.'''
        serializer = ProjectSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        project = serializer.save(owner=request.user)
        logger.info("Project created", extra={"project_id": project.id ,
                                               "user_id": request.user.id ,
                                               "project_name": project.name})
        return Response(ProjectSerializer(project).data, status=status.HTTP_201_CREATED)


# ── GET /api/projects/<id>/files/ ─────────────────────────────────────────────
class ProjectFilesView(APIView):
    '''GET /api/projects/<id>/files/'''

    def get(self, request, project_id: int):
        '''List all uploaded files for a given project.'''
        project = get_object_or_404(Project, pk=project_id, owner=request.user)
        files = project.files.all().order_by("-uploaded_at")
        return Response(UploadedFileSerializer(files, many=True).data)


# ── POST /api/projects/<id>/upload/ ──────────────────────────────────────────
class FileUploadView(APIView):
    """POST /api/projects/<id>/upload/"""

    def post(self, request, project_id: int):
        '''Upload files to a specific project.'''
        project = get_object_or_404(Project, pk=project_id, owner=request.user)

        uploaded_files = request.FILES.getlist("files")
        # e.g. ["myapp/views.py", "myapp/models.py"]
        relative_paths = request.POST.getlist("relative_paths")

        if not uploaded_files:
            return Response(
                {"error": "No files provided."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        saved_paths = []
        errors      = []

        for i, upload in enumerate(uploaded_files):
            # Get matching relative path if provided, else empty string
            rel_path = relative_paths[i] if i < len(relative_paths) else ""

            try:
                relative_stored, size = save_uploaded_file(
                    upload,
                    request.user.id,
                    project.pk,
                    relative_path=rel_path,   # ← pass relative path
                )
                UploadedFile.objects.create(
                    project=project,
                    original_filename=upload.name,
                    stored_path=relative_stored,
                    file_size_bytes=size,
                )
                saved_paths.append(
                    str(settings.UPLOAD_ROOT / relative_stored)
                )
            except Exception as exc:
                errors.append(f"{upload.name}: {exc}")

        if not saved_paths:
            return Response(
                {"errors": errors},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Index files into FAISS
        project.status = Project.Status.INDEXING
        project.save(update_fields=["status"])

        try:
            all_chunks = []
            for path in saved_paths:
                all_chunks.extend(chunk_file(path))

            if not all_chunks:
                raise ValueError("No parseable chunks found.")

            print(f"Total chunks before explanation: {len(all_chunks)}")
            # Step 1: get results (data + usage)
            results = generate_explanations_batch(all_chunks)

            # Step 2: split data and usage
            explanations = [item["data"] for item in results]
            usages = [item["usage"] for item in results]

            # Step 3: validate explanations (unchanged logic)
            explanations = _validate_explanations(explanations, all_chunks)
            logger.info("Generated explanations for uploaded files", extra={"project_id": project.pk, "chunk_count": len(all_chunks)})

            for chunk, exp in zip(all_chunks[:5], explanations[:5]):
                print(f"[GRAPH] {chunk.symbol_name} → {exp['dependencies']}")

            build_and_save_index(
                user_id=request.user.id,
                project_id=project.pk,
                chunks=all_chunks,
                explanations=explanations,
                usages=usages,
            )
            logger.info("FAISS index built and saved", extra={"project_id": project.pk})

            project.status        = Project.Status.READY
            project.error_message = ""
            project.save(update_fields=["status", "error_message"])

        except Exception as exc:
            logger.exception("Indexing failed for project %s", project.pk)
            project.status        = Project.Status.FAILED
            project.error_message = str(exc)
            project.save(update_fields=["status", "error_message"])
            return Response(
                {"error": "Indexing failed.", "detail": str(exc)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        return Response({
            "message":        "Files uploaded and indexed successfully.",
            "chunks_indexed": len(all_chunks),
            "files_saved":    len(saved_paths),
            "skipped_errors": errors,
        })



class ConversationListCreateView(APIView):
    """GET /api/projects/<id>/conversations/   POST (create new)"""

    def get(self, request, project_id: int):
        '''List all conversations for a given project.'''
        project = get_object_or_404(Project, pk=project_id, owner=request.user)
        convs   = project.conversations.all()
        return Response(ConversationListSerializer(convs, many=True).data)

    def post(self, request, project_id: int):
        '''Create a new conversation (chat) within a project.'''
        project = get_object_or_404(Project, pk=project_id, owner=request.user)
        title   = request.data.get("title", "New Chat").strip() or "New Chat"
        conv    = Conversation.objects.create(project=project, title=title)
        logger.info("Conversation created", extra={"conversation_id": conv.id, "project_id": project.pk, "user_id": request.user.id})
        return Response(ConversationSerializer(conv).data, status=status.HTTP_201_CREATED)


class ConversationDetailView(APIView):
    """GET /api/conversations/<id>/   PATCH (rename)   DELETE"""

    def _get_conv(self, request, conv_id):
        return get_object_or_404(
            Conversation, pk=conv_id, project__owner=request.user
        )

    def get(self, request, conv_id: int):
        '''Retrieve a specific conversation.'''
        conv = self._get_conv(request, conv_id)
        return Response(ConversationSerializer(conv).data)

    def patch(self, request, conv_id: int):
        '''Rename a specific conversation.'''
        conv  = self._get_conv(request, conv_id)
        title = request.data.get("title", "").strip()
        if title:
            conv.title = title
            conv.save(update_fields=["title"])
        return Response(ConversationSerializer(conv).data)

    def delete(self, request, conv_id: int):
        '''Delete a specific conversation.'''
        conv = self._get_conv(request, conv_id)
        conv.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class ConversationStreamView(APIView):
    """POST /api/conversations/<id>/stream/"""

    def post(self, request, conv_id: int):
        '''Stream the assistant's answer to a user query in real-time using Server-Sent Events (SSE).'''
        conv    = get_object_or_404(Conversation, pk=conv_id, project__owner=request.user)
        project = conv.project

        if project.status != Project.Status.READY:
            return Response({"error": f"Project not ready."}, status=409)

        serializer = QuerySerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        query = serializer.validated_data["query"]

        # ── Load short-term memory — last N turns from DB ─────────────────────
        SHORT_TERM_MEMORY_TURNS = 3

        recent_messages = conv.messages.order_by('-created_at')[: SHORT_TERM_MEMORY_TURNS * 2]
        # Reverse so they are in chronological order (oldest first)
        chat_history = [
            {"role": msg.role, "content": msg.content}
            for msg in reversed(recent_messages)
        ]

        # ── Save user message ─────────────────────────────────────────────────
        Message.objects.create(
            conversation=conv,
            role=Message.Role.USER,
            content=query,
        )
        conv.save(update_fields=["updated_at"])

        collected = {"full_text": "", "cited_chunks": [], 
        "usage": {"prompt_tokens": 0, "completion_tokens": 0}}

        print("Starting stream_answer with chat_history:", chat_history)
        
        def event_stream():
            # Pass chat_history into stream_answer
            for event in stream_answer(query, request.user.id, project.pk, chat_history):
                if event.startswith("data: "):
                    try:
                        payload = json.loads(event[6:])
                        if payload["type"] == "metadata":
                            collected["cited_chunks"] = payload["cited_chunks"]
                        elif payload["type"] == "done":
                            collected["full_text"] = payload["full_text"]
                            usage = payload.get("usage", {})
                            collected["usage"]["prompt_tokens"] = usage.get("prompt_tokens", 0)
                            collected["usage"]["completion_tokens"] = usage.get("completion_tokens", 0)
                    except Exception:
                        pass
                yield event

            # ── Save assistant reply ──────────────────────────────────────────
            answer = collected["full_text"]
            cited  = collected["cited_chunks"]
            if answer:
                Message.objects.create(
                    conversation=conv,
                    role=Message.Role.ASSISTANT,
                    content=answer,
                    cited_chunks=cited,
                    input_tokens=collected["usage"]["prompt_tokens"],     
                    output_tokens=collected["usage"]["completion_tokens"] 
                )
                if conv.messages.count() <= 2 and conv.title == "New Chat":
                    conv.title = query[:60] + ("…" if len(query) > 60 else "")
                    conv.save(update_fields=["title", "updated_at"])

        response = StreamingHttpResponse(event_stream(), content_type="text/event-stream")
        response["Cache-Control"]               = "no-cache"
        response["X-Accel-Buffering"]           = "no"
        response["Access-Control-Allow-Origin"] = "*"
        return response



class GithubRepoInfoView(APIView):
    """
    POST /api/github/info/
    Body: {"url": "https://github.com/owner/repo"}
    Returns repo name, description, file count preview.
    Used by frontend to pre-fill project name before importing.
    """

    def post(self, request):
        url = request.data.get('url', '').strip()
        if not url:
            return Response({'error': 'URL is required.'}, status=400)
        try:
            info = get_repo_info(url)
            logger.info("Retrieved GitHub repo info", extra={"repo_url": url})
            return Response(info)
        except GithubFetchError as e:
            logger.error("Failed to retrieve GitHub repo info", extra={"repo_url": url, "error": str(e)})
            return Response({'error': str(e)}, status=400)


class GithubImportView(APIView):
    """
    POST /api/projects/<id>/import-github/
    Body: {"url": "https://github.com/owner/repo"}

    Fetches all .py files from the GitHub repo,
    saves them to disk, and indexes them into FAISS —
    exactly like a manual file upload but sourced from GitHub.
    """

    def post(self, request, project_id: int):
        project = get_object_or_404(Project, pk=project_id, owner=request.user)
        url     = request.data.get('url', '').strip()

        if not url:
            return Response({'error': 'GitHub URL is required.'}, status=400)

        if project.status == Project.Status.INDEXING:
            return Response({'error': 'Project is already indexing.'}, status=409)

        # ── Fetch files from GitHub ───────────────────────────────────────────
        try:
            github_files = fetch_github_repo(url)
        except GithubFetchError as e:
            logger.error("Failed to fetch GitHub repo files", extra={"repo_url": url, "error": str(e)})
            return Response({'error': str(e)}, status=400)

        if not github_files:
            return Response({'error': 'No Python files found in repository.'}, status=400)

        # ── Save files to disk ────────────────────────────────────────────────
        saved_paths = []
        errors      = []

        for gf in github_files:
            try:
                # Build safe destination path
                base_dir = (
                    settings.UPLOAD_ROOT
                    / f"user_{request.user.id}"
                    / f"project_{project.pk}"
                )

                # Preserve folder structure from GitHub path
                parts       = Path(gf.path).parts
                folder_part = str(Path(*parts[:-1])) if len(parts) > 1 else ''
                dest_dir    = base_dir / folder_part if folder_part else base_dir
                dest_dir.mkdir(parents=True, exist_ok=True)

                # UUID filename for security
                import uuid
                safe_name = f"{uuid.uuid4().hex}.py"
                dest_path = dest_dir / safe_name
                dest_path.write_bytes(gf.content)

                relative = str(dest_path.relative_to(settings.UPLOAD_ROOT))

                UploadedFile.objects.create(
                    project=project,
                    original_filename=Path(gf.path).name,
                    stored_path=relative,
                    file_size_bytes=gf.size,
                )
                saved_paths.append(str(dest_path))

            except Exception as exc:
                errors.append(f"{gf.path}: {exc}")

        if not saved_paths:
            return Response(
                {'error': 'Failed to save any files.', 'details': errors},
                status=500
            )

        # ── Index into FAISS (same pipeline as manual upload) ─────────────────
        project.status = Project.Status.INDEXING
        project.github_url = url
        project.save(update_fields=['status', 'github_url'])

        try:
            all_chunks = []
            for path in saved_paths:
                all_chunks.extend(chunk_file(path))

            if not all_chunks:
                raise ValueError("No parseable Python chunks found in repository.")

            print(f"Total chunks before explanation: {len(all_chunks)}")
            # Step 1: get results (data + usage)
            results = generate_explanations_batch(all_chunks)

            # Step 2: split data and usage
            explanations = [item["data"] for item in results]
            usages = [item["usage"] for item in results]

            # Step 3: validate explanations (unchanged logic)
            explanations = _validate_explanations(explanations, all_chunks)
            logger.info("Generated explanations for GitHub-imported files", extra={"project_id": project.pk, "chunk_count": len(all_chunks)})

            for chunk, exp in zip(all_chunks[:5], explanations[:5]):
                print(f"[GRAPH] {chunk.symbol_name} → {exp['dependencies']}")

            build_and_save_index(
                user_id=request.user.id,
                project_id=project.pk,
                chunks=all_chunks,
                explanations=explanations,
                usages=usages,
            )
            logger.info("FAISS index built and saved", extra={"project_id": project.pk})

            project.status        = Project.Status.READY
            project.error_message = ''
            project.save(update_fields=['status', 'error_message'])

        except Exception as exc:
            logger.exception("GitHub indexing failed for project %s", project.pk)
            project.status        = Project.Status.FAILED
            project.error_message = str(exc)
            project.save(update_fields=['status', 'error_message'])
            return Response(
                {'error': 'Indexing failed.', 'detail': str(exc)},
                status=500
            )

        return Response({
            'message':        f'Successfully imported {len(saved_paths)} files from GitHub.',
            'files_imported': len(saved_paths),
            'chunks_indexed': len(all_chunks),
            'skipped_errors': errors,
        })


class GithubImportFilesView(APIView):
    """
    POST /api/projects/<id>/import-github-files/
    Step 1 — Downloads files from GitHub and saves to disk.
    Does NOT chunk, explain, or index. Just saves raw files.
    """
    def post(self, request, project_id: int):
        project = get_object_or_404(Project, pk=project_id, owner=request.user)
        url     = request.data.get('url', '').strip()
        if not url:
            return Response({'error': 'GitHub URL is required.'}, status=400)

        try:
            github_files = fetch_github_repo(url)
        except GithubFetchError as e:
            return Response({'error': str(e)}, status=400)

        if not github_files:
            return Response({'error': 'No Python files found.'}, status=400)

        # Save github_url to project
        project.github_url = url
        project.status     = Project.Status.PENDING
        project.save(update_fields=['github_url', 'status'])

        # Delete existing uploaded files for this project
        UploadedFile.objects.filter(project=project).delete()

        # Save files to disk
        saved_paths = []
        import uuid
        for gf in github_files:
            try:
                base_dir = (
                    settings.UPLOAD_ROOT
                    / f"user_{request.user.id}"
                    / f"project_{project.pk}"
                )
                parts       = Path(gf.path).parts
                folder_part = str(Path(*parts[:-1])) if len(parts) > 1 else ''
                dest_dir    = base_dir / folder_part if folder_part else base_dir
                dest_dir.mkdir(parents=True, exist_ok=True)
                safe_name   = f"{uuid.uuid4().hex}.py"
                dest_path   = dest_dir / safe_name
                dest_path.write_bytes(gf.content)

                UploadedFile.objects.create(
                    project=project,
                    original_filename=Path(gf.path).name,
                    original_path=gf.path,
                    stored_path=str(dest_path.relative_to(settings.UPLOAD_ROOT)),
                    file_size_bytes=gf.size,
                )
                saved_paths.append(str(dest_path))
            except Exception:
                continue

        return Response({
            'message':    f'Downloaded {len(saved_paths)} files from GitHub.',
            'files_saved': len(saved_paths),
        })


class IndexFilesView(APIView):
    """
    POST /api/projects/<id>/index-files/
    Step 2 — Chunks, explains, and indexes files already saved to disk.
    """
    def post(self, request, project_id: int):
        '''Index files that have already been saved to disk for this project.'''
        project = get_object_or_404(Project, pk=project_id, owner=request.user)

        if project.status == Project.Status.INDEXING:
            return Response({'error': 'Already indexing.'}, status=409)

        uploaded = UploadedFile.objects.filter(project=project)
        if not uploaded.exists():
            return Response(
                {'error': 'No files found. Import files first.'},
                status=400
            )

        project.status = Project.Status.INDEXING
        project.save(update_fields=['status'])

        try:
            all_chunks = []

            for f in uploaded:
                disk_path     = str(settings.UPLOAD_ROOT / f.stored_path)
                original_path = f.original_path or f.original_filename

                try:
                    chunks = chunk_file(disk_path)   # ← pass string, not tuple
                except Exception:
                    continue

                # Attach the stable original path to each chunk
                # so diff_chunks() can match them across re-imports
                for chunk in chunks:
                    chunk.original_path = original_path

                all_chunks.extend(chunks)

            if not all_chunks:
                raise ValueError("No parseable chunks found.")

            print(f"Total chunks before explanation: {len(all_chunks)}")
            # Step 1: get results (data + usage)
            results = generate_explanations_batch(all_chunks)

            # Step 2: split data and usage
            explanations = [item["data"] for item in results]
            usages = [item["usage"] for item in results]

            # Step 3: validate explanations (unchanged logic)
            explanations = _validate_explanations(explanations, all_chunks)
            logger.info("Generated explanations for files to be indexed", extra={"project_id": project.pk, "chunk_count": len(all_chunks)})

            for chunk, exp in zip(all_chunks[:5], explanations[:5]):
                print(f"[GRAPH] {chunk.symbol_name} → {exp['dependencies']}")

            build_and_save_index(
                user_id=request.user.id,
                project_id=project.pk,
                chunks=all_chunks,
                explanations=explanations,
                usages=usages,
                project=project,
            )
            logger.info("FAISS index built and saved", extra={"project_id": project.pk})

            project.status        = Project.Status.READY
            project.error_message = ''
            project.save(update_fields=['status', 'error_message'])

            return Response({
                'message':        'Indexing complete.',
                'files_indexed':  uploaded.count(),
                'chunks_indexed': len(all_chunks),
            })

        except Exception as e:
            project.status        = Project.Status.FAILED
            project.error_message = str(e)
            project.save(update_fields=['status', 'error_message'])
            return Response({'error': str(e)}, status=500)


class ReindexView(APIView):
    def post(self, request, project_id: int):
        project = get_object_or_404(Project, pk=project_id, owner=request.user)

        if not project.github_url:
            return Response({'error': 'No GitHub URL linked to this project.'}, status=400)

        if project.status == Project.Status.INDEXING:
            return Response({'error': 'Already indexing.'}, status=409)

        project.status = Project.Status.INDEXING
        project.save(update_fields=['status'])

        try:
            github_files = fetch_github_repo(project.github_url)

            # Save files and build (disk_path, original_path) pairs
            file_pairs = []
            import uuid
            for gf in github_files:
                base_dir = (
                    settings.UPLOAD_ROOT
                    / f"user_{request.user.id}"
                    / f"project_{project.pk}"
                )
                parts       = Path(gf.path).parts
                folder_part = str(Path(*parts[:-1])) if len(parts) > 1 else ''
                dest_dir    = base_dir / folder_part if folder_part else base_dir
                dest_dir.mkdir(parents=True, exist_ok=True)
                safe_name   = f"{uuid.uuid4().hex}.py"
                dest_path   = dest_dir / safe_name
                dest_path.write_bytes(gf.content)

                file_pairs.append((
                    str(dest_path),   # disk path — changes every time
                    gf.path,          # original GitHub path — always stable
                ))

            result = incremental_reindex(
                project=project,
                file_pairs=file_pairs,   # ← pairs not just paths
                user_id=request.user.id,
            )

            project.status = Project.Status.READY
            project.save(update_fields=['status'])

            logger.info("Re-indexing complete", extra={
                "project_id": project.pk,
                "added": result.added_count,
                "changed": result.changed_count,
                "deleted": result.deleted_count,
                "skipped": result.skipped_count,
            })

            return Response({
                'message': 'Re-indexing complete.',
                'added':   result.added_count,
                'changed': result.changed_count,
                'deleted': result.deleted_count,
                'skipped': result.skipped_count,
            })

        except Exception as e:
            project.status        = Project.Status.FAILED
            project.error_message = str(e)
            project.save(update_fields=['status', 'error_message'])
            return Response({'error': str(e)}, status=500)