"""
github_fetcher.py

Fetches Python files from a public GitHub repository using the
GitHub Contents API (no authentication required for public repos,
authenticated for private repos via GITHUB_TOKEN).

Supports:
  - Full repo download
  - Specific branch/tag/commit
  - Private repos via GITHUB_TOKEN env variable
  - Recursive traversal of all subdirectories
  - Skips __pycache__, .pyc, non-.py files automatically

Usage:
    files = fetch_github_repo("https://github.com/owner/repo")
    # Returns list of GithubFile(path, content, size)
"""

import re
import os
import requests
from dataclasses import dataclass
from django.conf import settings


@dataclass
class GithubFile:
    """A single Python file fetched from GitHub."""
    path:    str    # relative path inside repo e.g. "myapp/utils/helpers.py"
    content: bytes  # raw file bytes
    size:    int    # file size in bytes


class GithubFetchError(Exception):
    """Raised when GitHub API returns an error."""
    pass


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_github_url(url: str) -> tuple[str, str, str]:
    """
    Parse a GitHub URL into (owner, repo, ref).

    Supports:
        https://github.com/owner/repo
        https://github.com/owner/repo/tree/branch-name
        https://github.com/owner/repo/tree/v1.2.3
    
    Returns:
        (owner, repo, ref)   ref defaults to "HEAD" if not specified
    """
    url = url.strip().rstrip('/')

    if url.endswith('.git'):
        url = url[:-4]

    # Match with branch/tag: github.com/owner/repo/tree/ref
    m = re.match(
        r'https?://github\.com/([^/]+)/([^/]+)/tree/(.+)',
        url
    )
    if m:
        return m.group(1), m.group(2), m.group(3)

    # Match plain repo: github.com/owner/repo
    m = re.match(
        r'https?://github\.com/([^/]+)/([^/]+)',
        url
    )
    if m:
        return m.group(1), m.group(2), 'HEAD'

    raise GithubFetchError(
        f"Invalid GitHub URL: '{url}'. "
        "Expected format: https://github.com/owner/repo"
    )


def _get_headers() -> dict:
    """Build request headers. Includes auth token if GITHUB_TOKEN is set."""
    headers = {
        'Accept':     'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28',
    }
    token = getattr(settings, 'GITHUB_TOKEN', '') or os.environ.get('GITHUB_TOKEN', '')
    if token:
        headers['Authorization'] = f'Bearer {token}'
    return headers


def _api_get(url: str) -> dict | list:
    """Make a GET request to the GitHub API with error handling."""
    resp = requests.get(url, headers=_get_headers(), timeout=30)

    if resp.status_code == 404:
        raise GithubFetchError(
            "Repository not found. Make sure the URL is correct and the repo is public. "
            "For private repos, set GITHUB_TOKEN in your .env file."
        )
    if resp.status_code == 403:
        raise GithubFetchError(
            "GitHub API rate limit exceeded. "
            "Set GITHUB_TOKEN in your .env to get higher rate limits (5000 req/hr)."
        )
    if resp.status_code == 401:
        raise GithubFetchError(
            "GitHub authentication failed. Check your GITHUB_TOKEN in .env."
        )
    if not resp.ok:
        raise GithubFetchError(
            f"GitHub API error {resp.status_code}: {resp.text[:200]}"
        )

    return resp.json()


# ── Tree fetcher ───────────────────────────────────────────────────────────────

def _get_default_branch(owner: str, repo: str) -> str:
    """Get the actual default branch name (main, master, dev, etc.)"""
    try:
        data = _api_get(f"https://api.github.com/repos/{owner}/{repo}")
        return data.get('default_branch', 'main')
    except GithubFetchError:
        return 'main'

    
def _fetch_tree(owner: str, repo: str, ref: str) -> list[dict]:
    # ── Resolve HEAD to the actual default branch name ────────────────────────
    if ref == 'HEAD':
        ref = _get_default_branch(owner, repo)

    url  = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{ref}?recursive=1"
    data = _api_get(url)

    if data.get('truncated'):
        raise GithubFetchError(
            "Repository is too large (>100,000 files)."
        )

    return data.get('tree', [])


def _fetch_file_content(owner: str, repo: str, path: str, ref: str) -> bytes:
    """
    Download the raw content of a single file.
    Uses the raw content endpoint for efficiency (no base64 decoding needed).
    """
    url  = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"
    resp = requests.get(url, headers=_get_headers(), timeout=30)
    if not resp.ok:
        raise GithubFetchError(f"Failed to download {path}: HTTP {resp.status_code}")
    return resp.content


# ── Public API ────────────────────────────────────────────────────────────────

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024   # 10 MB — same limit as manual uploads
SKIP_DIRS = {'__pycache__', '.git', 'node_modules', '.venv', 'venv', 'env',
             '.tox', 'dist', 'build', 'eggs', '.eggs', 'htmlcov', '.mypy_cache'}


def fetch_github_repo(github_url: str) -> list[GithubFile]:
    """
    Fetch all Python files from a GitHub repository.

    Args:
        github_url: Full GitHub URL, e.g.:
                    "https://github.com/django/django"
                    "https://github.com/django/django/tree/stable/5.0.x"

    Returns:
        List of GithubFile objects (path, content, size).
        Only .py files are returned. __pycache__, venv, .git etc. are skipped.

    Raises:
        GithubFetchError: if the repo doesn't exist, is private without a token,
                          or rate limits are exceeded.
    """
    owner, repo, ref = _parse_github_url(github_url)

    # Fetch the full file tree in one API call
    tree = _fetch_tree(owner, repo, ref)

    # Filter to .py files only, skipping ignored directories
    py_items = []
    for item in tree:
        if item.get('type') != 'blob':
            continue
        path = item['path']

        # Skip ignored directories
        parts = path.split('/')
        if any(part in SKIP_DIRS for part in parts):
            continue

        # .py files only
        if not path.endswith('.py'):
            continue

        # Skip files that are too large
        size = item.get('size', 0)
        if size > MAX_FILE_SIZE_BYTES:
            continue

        py_items.append((path, size))

    if not py_items:
        raise GithubFetchError(
            "No Python (.py) files found in this repository. "
            "Make sure the repo contains Python code."
        )

    # Download file contents
    files = []
    for path, size in py_items:
        try:
            content = _fetch_file_content(owner, repo, path, ref)
            files.append(GithubFile(path=path, content=content, size=len(content)))
        except GithubFetchError:
            # Skip files that fail to download individually
            continue

    return files


def get_repo_info(github_url: str) -> dict:
    owner, repo, ref = _parse_github_url(github_url)
    try:
        data = _api_get(f"https://api.github.com/repos/{owner}/{repo}")
        # Use actual default branch instead of HEAD
        actual_ref = ref if ref != 'HEAD' else data.get('default_branch', 'main')
        return {
            'name':        data.get('name', repo),
            'description': data.get('description', ''),
            'stars':       data.get('stargazers_count', 0),
            'language':    data.get('language', 'Python'),
            'owner':       owner,
            'ref':         actual_ref,   # ← shows "main" instead of "HEAD"
        }
    except GithubFetchError:
        return {
            'name': repo, 'description': '', 'stars': 0,
            'language': 'Python', 'owner': owner, 'ref': ref
        }