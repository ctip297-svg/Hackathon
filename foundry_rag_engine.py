"""
Foundry-compatible RAG backend for IMAC immunisation advisor.
Uses Microsoft Foundry for model inference (chat completions and embeddings).
Compatible with Streamlit app via the same generate_response() interface.
"""

import os
import re
import time
import uuid
import hashlib
from io import BytesIO
from typing import List, Dict, Tuple
import json
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from PyPDF2 import PdfReader

# Load environment variables
load_dotenv()


def warn_placeholder_config(foundry_endpoint, foundry_key, foundry_chat_model, foundry_embedding_model):
    """Warn at startup when .env contains placeholder Foundry credentials."""
    warnings = []

    if not foundry_endpoint or "your-foundry.azure.com" in foundry_endpoint:
        warnings.append("Foundry endpoint is not configured or is still using the placeholder value.")
    if not foundry_key or foundry_key.startswith("your-"):
        warnings.append("Foundry API key is not configured or is still using the placeholder value.")
    if not foundry_chat_model or foundry_chat_model.startswith("your-"):
        warnings.append("Foundry chat model name is not configured or is still using the placeholder value.")
    if not foundry_embedding_model or foundry_embedding_model.startswith("your-"):
        warnings.append("Foundry embedding model name is not configured or is still using the placeholder value.")

    if warnings:
        print("WARNING: Invalid Foundry configuration detected in .env. Please update the following settings:")
        for warning in warnings:
            print(f" - {warning}")
        print("The app will fall back to mock mode unless valid Foundry credentials are provided.")


# Retrieve Foundry Variables
foundry_endpoint = os.getenv("FOUNDRY_ENDPOINT")
foundry_api_key = os.getenv("FOUNDRY_API_KEY")
foundry_chat_model = os.getenv("FOUNDRY_CHAT_MODEL", "gpt-5-mini")
foundry_embedding_model = os.getenv("FOUNDRY_EMBEDDING_MODEL", "text-embedding-3-small")
foundry_api_version = os.getenv("FOUNDRY_API_VERSION", "2025-08-07")

VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "./chroma_store")
VECTOR_COLLECTION_NAME = os.getenv("VECTOR_COLLECTION_NAME", "imac_guidance")

warn_placeholder_config(foundry_endpoint, foundry_api_key, foundry_chat_model, foundry_embedding_model)

# Security configuration
MAX_USER_PROMPT_LENGTH = 1000
MAX_SEARCH_RESULTS = 3
MAX_VECTOR_RESULTS = 3
MAX_CONVERSATION_TURNS = 10
MAX_CONTEXT_CHUNKS = 3
MAX_CONTEXT_CHUNK_CHARS = 700
MAX_CONTEXT_TOTAL_CHARS = 2500
MAX_FOUNDARY_OUTPUT_TOKENS = 1024
MAX_FOUNDARY_RETRY_OUTPUT_TOKENS = 2048


def _debug_log(message: str):
    if os.getenv("RAG_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}:
        print(f"DEBUG: {message}")


# Data sources for ingestion
SOURCE_URLS = [
    "./vaccines.json",
    "https://immune.org.nz",
    "https://www.health.govt.nz/publication/nz-immunisation-handbook-2023",
    "https://immune.org.nz/immunisation/programmes/national-immunisation-schedule",
    "https://www.tewhatuora.govt.nz/for-health-professionals/clinical-guidance/immunisation-handbook",
    "https://static.info.content.health.nz/docs/health-pros/topics/immunisations/immunisation-handbook-2026-v2.pdf",
]

HEADERS = {
    "User-Agent": "IMAC-RAG-Agent/1.0 (+https://immune.org.nz)"
}


class FoundryClient:
    """Wrapper for Foundry inference API calls (chat completions and embeddings)."""
    
    def __init__(self, endpoint: str, api_key: str, api_version: str = foundry_api_version):
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.api_version = api_version
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.parsed_endpoint = urlparse(self.endpoint)
        self.service_root = f"{self.parsed_endpoint.scheme}://{self.parsed_endpoint.netloc}"
        self.use_responses_api = self.endpoint.endswith("/responses") or "/responses/" in self.endpoint
        self.base_endpoint = self.endpoint
        if self.use_responses_api:
            self.base_endpoint = self.endpoint[: self.endpoint.rfind("/responses")]

        if self.use_responses_api:
            self.embeddings_url = f"{self.service_root}/openai/v1/embeddings"
        elif self.endpoint.endswith("/embeddings"):
            self.embeddings_url = self.endpoint
        elif self.endpoint.endswith("/openai/v1"):
            self.embeddings_url = f"{self.endpoint}/embeddings"
        elif "/openai/v1/" in self.endpoint:
            self.embeddings_url = re.sub(r"/openai/v1/.*$", "/openai/v1/embeddings", self.endpoint)
        else:
            self.embeddings_url = f"{self.endpoint}/openai/v1/embeddings"
    
    def _build_url(self, path: str) -> str:
        if path == "embeddings":
            return self.embeddings_url

        if self.use_responses_api:
            if path == "responses":
                return self.endpoint
            return f"{self.base_endpoint}/{path}"
        return f"{self.endpoint}/{path}"

    def _request(self, path: str, payload: dict):
        url = self._build_url(path)
        if self.use_responses_api or "/openai/v1/" in self.endpoint:
            params = {}
        else:
            params = {"api-version": self.api_version} if self.api_version else {}

        response = requests.post(url, headers=self.headers, params=params, json=payload, timeout=30)
        if response.status_code >= 400:
            if response.status_code == 404 and path == "embeddings" and self.use_responses_api:
                fallback_urls = [
                    f"{self.base_endpoint}/{path}",
                    f"{self.endpoint}/{path}",
                ]
                for fallback_url in fallback_urls:
                    if fallback_url == url:
                        continue
                    print(f"Warning: embeddings endpoint 404; trying fallback {fallback_url}")
                    alt_response = requests.post(fallback_url, headers=self.headers, params=params, json=payload, timeout=30)
                    if alt_response.status_code < 400:
                        return alt_response.json()
            raise RuntimeError(f"Foundry API error {response.status_code}: {response.text or response.reason} (url={url})")
        return response.json()
    
    def chat_completions_create(self, model: str, messages: List[Dict[str, str]], max_tokens: int = 500, temperature: float = 0.0):
        """Call Foundry chat completions API."""
        if self.use_responses_api:
            serialized_input = "\n".join(msg["content"] for msg in messages)
            payload = {
                "model": model,
                "input": serialized_input,
                "max_output_tokens": max_tokens,
                "reasoning": {"effort": "low"},
                "text": {"verbosity": "low"},
            }
            return self._request("responses", payload)

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        return self._request("chat/completions", payload)
    
    def _extract_embeddings(self, response: dict) -> List[List[float]]:
        if not isinstance(response, dict):
            raise ValueError("Embeddings response is not a dict")

        if "data" in response and isinstance(response["data"], list):
            embeddings = []
            for item in response["data"]:
                if isinstance(item, dict) and "embedding" in item:
                    embeddings.append(item["embedding"])
            if embeddings:
                return embeddings

        if "output" in response and isinstance(response["output"], list):
            embeddings = []
            for item in response["output"]:
                if isinstance(item, dict) and "embedding" in item:
                    embeddings.append(item["embedding"])
            if embeddings:
                return embeddings

        if "embeddings" in response and isinstance(response["embeddings"], list):
            embeddings = [item for item in response["embeddings"] if isinstance(item, list)]
            if embeddings:
                return embeddings

        raise KeyError("Unable to extract embeddings from Foundry response")

    def embeddings_create(self, model: str, input_texts: List[str]):
        """Call Foundry embeddings API."""
        payload = {
            "model": model,
            "input": input_texts,
        }
        path = "embeddings" if self.use_responses_api else "embeddings"
        response = self._request(path, payload)
        return {"data": [{"embedding": emb} for emb in self._extract_embeddings(response)]}


# Initialize Foundry Client
foundry_client = None
if foundry_endpoint and foundry_api_key and "your-foundry.azure.com" not in foundry_endpoint:
    try:
        foundry_client = FoundryClient(
            endpoint=foundry_endpoint,
            api_key=foundry_api_key,
            api_version=foundry_api_version,
        )
    except Exception as e:
        print(f"Failed to initialize Foundry client: {e}")


def create_vector_store(persist_directory: str = VECTOR_STORE_DIR) -> chromadb.api.client.Client:
    return chromadb.Client(
        Settings(
            persist_directory=persist_directory,
            is_persistent=True,
        )
    )


def get_collection_name(collection_item):
    """Extract collection name from either dict or Collection object."""
    if isinstance(collection_item, dict):
        return collection_item.get("name")
    return getattr(collection_item, "name", None)


def vector_collection_exists(collection_name: str = VECTOR_COLLECTION_NAME, persist_directory: str = VECTOR_STORE_DIR) -> bool:
    client = create_vector_store(persist_directory)
    return collection_name in [get_collection_name(collection) for collection in client.list_collections()]


def vector_collection_has_data(collection_name: str = VECTOR_COLLECTION_NAME, persist_directory: str = VECTOR_STORE_DIR) -> bool:
    client = create_vector_store(persist_directory)
    if collection_name not in [get_collection_name(collection) for collection in client.list_collections()]:
        return False

    collection = client.get_collection(collection_name)
    try:
        return collection.count() > 0
    except Exception:
        documents = collection.get().get('documents', [])
        return bool(documents)


def fetch_page(url: str, timeout: int = 20) -> str:
    response = requests.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.text


def fetch_pdf_text(url: str, timeout: int = 30) -> str:
    response = requests.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    reader = PdfReader(BytesIO(response.content))
    pages: List[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def extract_visible_text_from_html(html: str) -> Tuple[str, List[str]]:
    soup = BeautifulSoup(html, "html.parser")
    
    # Remove script and style tags to avoid parsing them
    for tag in soup.find_all(["script", "style", "nav", "footer"]):
        tag.decompose()
    
    title = (soup.title.string or "").strip() if soup.title else "IMAC Guidance"

    candidates: List[str] = []
    
    # Primary: try semantic content tags
    for selector in ["article", "main", "section[role='main']", "div[role='main']"]:
        for element in soup.select(selector):
            text = element.get_text(separator="\n", strip=True)
            if len(text) > 120:
                candidates.append(text)
    
    # Secondary: if semantic tags found minimal content, extract paragraphs and headings
    if len(candidates) < 2:
        for element in soup.find_all(["p", "h1", "h2", "h3", "h4", "li"]):
            text = element.get_text(separator="\n", strip=True)
            if len(text) >= 30:  # Lower threshold for individual elements
                candidates.append(text)
    
    # Tertiary: if still minimal, grab substantial divs
    if len(candidates) < 3:
        for div in soup.find_all("div", class_=None):
            text = div.get_text(separator="\n", strip=True)
            if 200 <= len(text) <= 2000:
                candidates.append(text)

    cleaned: List[str] = []
    for fragment in candidates:
        normalized = re.sub(r"\s+", " ", fragment).strip()
        if len(normalized) >= 50:  # Slightly relaxed threshold
            cleaned.append(normalized)

    return title or "IMAC Guidance", cleaned


def chunk_text(text: str, chunk_size: int = 250, overlap: int = 50) -> List[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [text.strip()]

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap
    return chunks


def build_documents_from_url(url: str) -> List[Dict[str, str]]:
    local_path = os.path.abspath(url) if os.path.exists(url) else None
    if local_path:
        lower_path = local_path.lower()
        if lower_path.endswith(".json"):
            return build_documents_from_local_json(local_path)
        if lower_path.endswith((".txt", ".md")):
            with open(local_path, "r", encoding="utf-8") as handle:
                text = handle.read()
            title = os.path.basename(local_path)
            documents: List[Dict[str, str]] = []
            for chunk in chunk_text(text):
                chunk_hash = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                documents.append(
                    {
                        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{local_path}#{chunk_hash}")),
                        "title": title,
                        "content": chunk,
                        "url": local_path,
                        "source": local_path,
                        "section": title,
                    }
                )
            print(f"Built {len(documents)} chunk documents from local path {local_path}")
            return documents

    if url.lower().endswith(".pdf"):
        raw_text = fetch_pdf_text(url)
        title = "NZ Immunisation Handbook PDF"
        text_blocks = [raw_text]
    else:
        html = fetch_page(url)
        title, text_blocks = extract_visible_text_from_html(html)

    documents: List[Dict[str, str]] = []
    for section_index, block in enumerate(text_blocks, start=1):
        for chunk in chunk_text(block):
            chunk_hash = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
            documents.append(
                {
                    "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{url}#{section_index}-{chunk_hash}")),
                    "title": title,
                    "content": chunk,
                    "url": url,
                    "source": url,
                    "section": title,
                }
            )
    print(f"Built {len(documents)} chunk documents from {url}")
    return documents


def extract_blob_text(blob_name: str, blob_bytes: bytes) -> str:
    lower_name = blob_name.lower()

    if lower_name.endswith(".pdf"):
        reader = PdfReader(BytesIO(blob_bytes))
        pages: List[str] = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)

    if lower_name.endswith((".html", ".htm")):
        html = blob_bytes.decode("utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup.find_all(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)

    if lower_name.endswith((".txt", ".md", ".json", ".csv")):
        return blob_bytes.decode("utf-8", errors="ignore")

    return blob_bytes.decode("utf-8", errors="ignore")


def build_documents_from_blob(blob_name: str, blob_bytes: bytes, blob_url: str) -> List[Dict[str, str]]:
    text = extract_blob_text(blob_name, blob_bytes)
    if not text.strip():
        return []

    documents: List[Dict[str, str]] = []
    for chunk in chunk_text(text):
        chunk_hash = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
        documents.append(
            {
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{blob_url}#{chunk_hash}")),
                "title": blob_name,
                "content": chunk,
                "url": blob_url,
                "source": blob_url,
                "section": blob_name,
            }
        )

    print(f"Built {len(documents)} chunk documents from blob {blob_name}")
    return documents


def flatten_json_value(value, prefix: str = "") -> List[str]:
    if isinstance(value, dict):
        flattened = []
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.extend(flatten_json_value(item, next_prefix))
        return flattened
    if isinstance(value, list):
        flattened = []
        for index, item in enumerate(value):
            next_prefix = f"{prefix}[{index}]" if prefix else f"[{index}]"
            flattened.extend(flatten_json_value(item, next_prefix))
        return flattened
    return [f"{prefix}: {value}" if prefix else str(value)]


def build_documents_from_local_json(file_path: str) -> List[Dict[str, str]]:
    with open(file_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    flattened_segments = flatten_json_value(data)
    text = "\n".join(flattened_segments)
    if not text.strip():
        return []

    documents: List[Dict[str, str]] = []
    for chunk in chunk_text(text):
        chunk_hash = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
        documents.append(
            {
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{file_path}#{chunk_hash}")),
                "title": os.path.basename(file_path),
                "content": chunk,
                "url": file_path,
                "source": file_path,
                "section": os.path.basename(file_path),
            }
        )

    print(f"Built {len(documents)} chunk documents from local JSON file {file_path}")
    return documents


def embed_texts(texts: List[str], batch_size: int = 16) -> List[List[float]]:
    if foundry_client is None:
        raise RuntimeError("Foundry client is not configured for embeddings.")

    embeddings: List[List[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        try:
            response = foundry_client.embeddings_create(
                model=foundry_embedding_model,
                input_texts=batch,
            )
            # Extract embeddings from Foundry API response
            for item in response.get("data", []):
                embeddings.append(item["embedding"])
            time.sleep(0.2)
        except Exception as e:
            print(f"Warning: embedding batch failed: {e}")
            # Return dummy embeddings on failure
            embeddings.extend([[0.0] * 1536 for _ in batch])
    return embeddings


def add_documents_to_collection(
    collection,
    ids: List[str],
    contents: List[str],
    metadatas: List[Dict[str, str]],
    embeddings: List[List[float]],
    batch_size: int = 100,
) -> int:
    total_added = 0
    for start in range(0, len(ids), batch_size):
        batch_ids = ids[start : start + batch_size]
        batch_contents = contents[start : start + batch_size]
        batch_metadatas = metadatas[start : start + batch_size]
        batch_embeddings = embeddings[start : start + batch_size]
        collection.add(
            ids=batch_ids,
            documents=batch_contents,
            metadatas=batch_metadatas,
            embeddings=batch_embeddings,
        )
        total_added += len(batch_ids)
    return total_added


def ingest_sources(collection_name: str = VECTOR_COLLECTION_NAME, persist_directory: str = VECTOR_STORE_DIR) -> int:
    if foundry_client is None:
        raise RuntimeError("Foundry client is not configured. Cannot ingest vector data.")

    client = create_vector_store(persist_directory)
    existing = [get_collection_name(collection) for collection in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(collection_name)

    collection = client.get_or_create_collection(name=collection_name)

    documents: List[Dict[str, str]] = []
    for url in SOURCE_URLS:
        try:
            documents.extend(build_documents_from_url(url))
            time.sleep(1.0)
        except Exception as exc:
            print(f"Warning: failed to ingest {url}: {exc}")

    if not documents:
        print("No documents were created for ingestion.")
        return 0

    contents = [doc["content"] for doc in documents]
    embeddings = embed_texts(contents)
    ids = [doc["id"] for doc in documents]
    metadatas = [
        {
            "title": doc["title"],
            "url": doc["url"],
            "source": doc["source"],
            "section": doc["section"],
        }
        for doc in documents
    ]

    indexed_count = add_documents_to_collection(collection, ids, contents, metadatas, embeddings)

    print(f"Indexed {indexed_count} chunk embeddings into collection '{collection_name}'.")
    return indexed_count


def ingest_blob_container(
    container_name: str = None,
    prefix: str = None,
    collection_name: str = VECTOR_COLLECTION_NAME,
    persist_directory: str = VECTOR_STORE_DIR,
    connection_string: str = None,
) -> int:
    if foundry_client is None:
        raise RuntimeError("Foundry client is not configured. Cannot ingest vector data.")

    try:
        from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
    except ImportError as exc:
        raise RuntimeError("The azure-storage-blob package is required for blob ingestion. Install it with 'pip install azure-storage-blob'.") from exc

    resolved_connection_string = connection_string or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    resolved_container_name = container_name or os.getenv("AZURE_STORAGE_CONTAINER_NAME") or os.getenv("AZURE_BLOB_CONTAINER_NAME")

    if not resolved_connection_string:
        raise RuntimeError("Azure Blob ingestion requires AZURE_STORAGE_CONNECTION_STRING or a connection_string argument.")
    if not resolved_container_name:
        raise RuntimeError("Azure Blob ingestion requires AZURE_STORAGE_CONTAINER_NAME, AZURE_BLOB_CONTAINER_NAME, or a container_name argument.")

    client = create_vector_store(persist_directory)
    existing = [get_collection_name(collection) for collection in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(collection_name)

    collection = client.get_or_create_collection(name=collection_name)

    service_client = BlobServiceClient.from_connection_string(resolved_connection_string)
    container_client = service_client.get_container_client(resolved_container_name)

    documents: List[Dict[str, str]] = []
    blob_iter = container_client.list_blobs(name_starts_with=prefix) if prefix else container_client.list_blobs()
    for blob in blob_iter:
        try:
            blob_client = container_client.get_blob_client(blob.name)
            blob_bytes = blob_client.download_blob().readall()
            documents.extend(build_documents_from_blob(blob.name, blob_bytes, blob_client.url))
            time.sleep(0.2)
        except Exception as exc:
            print(f"Warning: failed to ingest blob {blob.name}: {exc}")

    if not documents:
        print("No documents were created from blob ingestion.")
        return 0

    contents = [doc["content"] for doc in documents]
    embeddings = embed_texts(contents)
    ids = [doc["id"] for doc in documents]
    metadatas = [
        {
            "title": doc["title"],
            "url": doc["url"],
            "source": doc["source"],
            "section": doc["section"],
        }
        for doc in documents
    ]

    indexed_count = add_documents_to_collection(collection, ids, contents, metadatas, embeddings)

    print(f"Indexed {indexed_count} chunk embeddings from blob container '{resolved_container_name}' into collection '{collection_name}'.")
    return indexed_count


def ingest_local_json_file(file_path: str, collection_name: str = VECTOR_COLLECTION_NAME, persist_directory: str = VECTOR_STORE_DIR) -> int:
    if foundry_client is None:
        raise RuntimeError("Foundry client is not configured. Cannot ingest vector data.")

    client = create_vector_store(persist_directory)
    existing = [get_collection_name(collection) for collection in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(collection_name)

    collection = client.get_or_create_collection(name=collection_name)

    documents = build_documents_from_local_json(file_path)
    if not documents:
        print(f"No documents were created from local JSON file {file_path}.")
        return 0

    contents = [doc["content"] for doc in documents]
    embeddings = embed_texts(contents)
    ids = [doc["id"] for doc in documents]
    metadatas = [
        {
            "title": doc["title"],
            "url": doc["url"],
            "source": doc["source"],
            "section": doc["section"],
        }
        for doc in documents
    ]

    indexed_count = add_documents_to_collection(collection, ids, contents, metadatas, embeddings)

    print(f"Indexed {indexed_count} chunk embeddings from local JSON file '{file_path}' into collection '{collection_name}'.")
    return indexed_count


def normalize_query_results(results):
    """Handle various Chroma query result shapes."""
    if isinstance(results, dict):
        documents = results.get("documents", [[]])
        metadatas = results.get("metadatas", [[]])
    else:
        documents = getattr(results, "documents", [[]])
        metadatas = getattr(results, "metadatas", [[]])

    documents = documents[0] if documents and isinstance(documents[0], list) else documents
    metadatas = metadatas[0] if metadatas and isinstance(metadatas[0], list) else metadatas
    return documents, metadatas


def query_vector_store(question: str, collection_name: str = VECTOR_COLLECTION_NAME, persist_directory: str = VECTOR_STORE_DIR) -> Tuple[List[str], List[str]]:
    client = create_vector_store(persist_directory)
    if collection_name not in [get_collection_name(collection) for collection in client.list_collections()]:
        return [], []

    collection = client.get_collection(collection_name)
    
    try:
        response = foundry_client.embeddings_create(
            model=foundry_embedding_model,
            input_texts=[question],
        )
        query_embedding = response["data"][0]["embedding"]
    except Exception as e:
        print(f"Embedding generation failed: {e}")
        return [], []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=MAX_VECTOR_RESULTS,
        include=["documents", "metadatas", "distances"],
    )

    documents, metadatas = normalize_query_results(results)
    chunks: List[str] = []
    citations: List[str] = []
    for doc, metadata in zip(documents, metadatas):
        chunks.append(doc)
        source = metadata.get("source") or metadata.get("url") or "Official IMAC Guidance"
        if source not in citations:
            citations.append(source)

    return chunks, citations


def ensure_vector_index(persist_directory: str = VECTOR_STORE_DIR, collection_name: str = VECTOR_COLLECTION_NAME) -> bool:
    if vector_collection_has_data(collection_name, persist_directory):
        client = create_vector_store(persist_directory)
        collection = client.get_collection(collection_name)
        try:
            count = collection.count()
        except Exception:
            count = len(collection.get().get("documents", []))
        print(f"Vector collection '{collection_name}' already exists with {count} documents.")
        return True

    if vector_collection_exists(collection_name, persist_directory):
        print(f"Existing vector collection '{collection_name}' is empty or invalid. Recreating it.")
        client = create_vector_store(persist_directory)
        client.delete_collection(collection_name)

    try:
        ingest_sources(collection_name, persist_directory)
        return vector_collection_has_data(collection_name, persist_directory)
    except Exception as e:
        print(f"Failed to create vector index: {e}")
        return False


def get_mock_response(user_prompt):
    """
    Provides mock responses for development when Foundry is not configured.
    """
    prompt_lower = user_prompt.lower()

    if "vaccine schedule" in prompt_lower or "schedule" in prompt_lower:
        return (
            "The standard vaccine schedule for children in New Zealand follows the National Immunisation Schedule. Key milestones include:\n\n"
            "- 6 weeks: DTaP-IPV-HepB/Hib, PCV13, Rotavirus\n"
            "- 3 months: DTaP-IPV-HepB/Hib, PCV13, Rotavirus\n"
            "- 5 months: DTaP-IPV-HepB/Hib, PCV13\n"
            "- 12 months: MMR, PCV13, Meningococcal B\n"
            "- 15 months: MMR, Varicella\n\n"
            "Please consult the latest IMAC guidelines for complete details.",
            ["[NZ Immunisation Handbook 2023](https://www.health.govt.nz/publication/nz-immunisation-handbook-2023)"]
        )
    elif "catch-up" in prompt_lower or "missed" in prompt_lower:
        return (
            "**Catch-up Vaccination Guidelines:**\n\n"
            "- Give vaccines appropriate for current age, regardless of when previous doses were given\n"
            "- Minimum intervals must still be observed between doses\n"
            "- No need to restart the series if the interval has been exceeded\n\n"
            "Refer to the NZ Immunisation Handbook for specific catch-up schedules by age.",
            ["[NZ Immunisation Handbook - Catch-up Schedules](https://www.health.govt.nz/publication/nz-immunisation-handbook-2023)"]
        )
    else:
        return (
            "This is a secure development mock response. To get real IMAC guidance, please configure your Foundry credentials in the .env file and build the local vector store.\n\n"
            "The question you asked would normally be answered using official IMAC guidelines retrieved from our knowledge base.",
            ["[IMAC Website](https://immune.org.nz)"]
        )


def trim_context_chunks(retrieved_chunks: List[str], max_chunks: int = MAX_CONTEXT_CHUNKS, max_chunk_chars: int = MAX_CONTEXT_CHUNK_CHARS, max_total_chars: int = MAX_CONTEXT_TOTAL_CHARS) -> List[str]:
    trimmed_chunks: List[str] = []
    total_chars = 0

    for chunk in retrieved_chunks[:max_chunks]:
        normalized = re.sub(r"\s+", " ", chunk).strip()
        if len(normalized) > max_chunk_chars:
            normalized = normalized[:max_chunk_chars].rsplit(" ", 1)[0].strip()
            if not normalized:
                normalized = normalized[:max_chunk_chars]
            normalized = f"{normalized.strip()}..."

        if total_chars + len(normalized) > max_total_chars and trimmed_chunks:
            break

        trimmed_chunks.append(normalized)
        total_chars += len(normalized)

    return trimmed_chunks


def generate_response(user_prompt, conversation_history=None):
    user_prompt = user_prompt.strip()
    if not user_prompt:
        return "Security Alert: Empty query received. Please provide a valid clinical question.", []

    if len(user_prompt) > MAX_USER_PROMPT_LENGTH:
        return f"Security Alert: Query exceeds maximum allowed length ({MAX_USER_PROMPT_LENGTH} characters).", []

    if not foundry_client:
        return get_mock_response(user_prompt)

    conversation_history = conversation_history or []
    history_messages = []
    for message in conversation_history[-MAX_CONVERSATION_TURNS:]:
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        stripped_content = content.strip()
        if stripped_content:
            history_messages.append({"role": role, "content": stripped_content})

    system_message = {
        "role": "system",
        "content": (
            "You are an IMAC immunisation advisor. Use the earlier turns in the conversation to preserve context, "
            "resolve follow-up references, and connect the current answer to prior discussion."
        ),
    }

    def build_chat_messages(prompt_body: str) -> List[Dict[str, str]]:
        messages = [system_message]
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt_body})
        return messages

    def _extract_response_text(api_response: dict) -> str:
        def _extract_from_content(content):
            texts = []
            if isinstance(content, str):
                return [content]
            if isinstance(content, dict):
                if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                    return [content["text"]]
                if isinstance(content.get("text"), str):
                    return [content["text"]]
                if isinstance(content.get("content"), str):
                    return [content["content"]]
                if isinstance(content.get("content"), list):
                    for item in content["content"]:
                        texts.extend(_extract_from_content(item))
                return texts
            if isinstance(content, list):
                for item in content:
                    texts.extend(_extract_from_content(item))
                return texts
            return texts

        def _collect_candidate_texts(value):
            candidates = []
            if isinstance(value, str):
                stripped = value.strip()
                if len(stripped) > 30 and " " in stripped:
                    candidates.append(stripped)
                return candidates
            if isinstance(value, dict):
                for key, subvalue in value.items():
                    if key in {"text", "output_text", "content"}:
                        candidates.extend(_collect_candidate_texts(subvalue))
                return candidates
            if isinstance(value, list):
                for item in value:
                    candidates.extend(_collect_candidate_texts(item))
                return candidates
            return candidates

        if not isinstance(api_response, dict):
            raise ValueError("API response is not a dict")

        if "choices" in api_response:
            choice = api_response["choices"][0]
            if isinstance(choice, dict) and "message" in choice:
                return choice["message"].get("content", "").strip()

        if "output_text" in api_response and isinstance(api_response["output_text"], str):
            return api_response["output_text"].strip()

        output = api_response.get("output")
        if isinstance(output, list):
            texts = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "message":
                    texts.extend(_extract_from_content(item.get("content")))
                elif isinstance(item.get("text"), str):
                    texts.append(item["text"])
            filtered = [t.strip() for t in texts if t and t.strip()]
            if filtered:
                return "\n".join(filtered).strip()

        if "text" in api_response and isinstance(api_response["text"], dict):
            nested = api_response["text"].get("text")
            if isinstance(nested, str) and nested.strip():
                return nested.strip()

        candidates = _collect_candidate_texts(api_response)
        if candidates:
            return candidates[0]

        _debug_log(f"Unable to extract text from Foundry response: {json.dumps(api_response, indent=2)[:3000]}")
        raise KeyError("Unable to extract text from Foundry response")

    def _call_foundry(messages: List[Dict[str, str]], max_tokens: int):
        response = foundry_client.chat_completions_create(
            model=foundry_chat_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
        )

        if (
            isinstance(response, dict)
            and response.get("status") == "incomplete"
            and response.get("incomplete_details", {}).get("reason") == "max_output_tokens"
        ):
            _debug_log("Foundry response was incomplete due to max_output_tokens; retrying with larger output budget.")
            response = foundry_client.chat_completions_create(
                model=foundry_chat_model,
                messages=messages,
                max_tokens=MAX_FOUNDARY_RETRY_OUTPUT_TOKENS,
                temperature=0.0,
            )

        return response

    def condense_query(prompt: str, history: list) -> str:
        if not history:
            return prompt
        condensation_prompt = (
            "Given the following conversation history and the user's follow-up question, "
            "rephrase the follow-up question to be a standalone query that can be used "
            "to search a medical database. Do not answer it, just rephrase it.\n\n"
            f"Chat History:\n{history}\n\n"
            f"Follow-up Question: {prompt}"
        )
        try:
            response = _call_foundry([{"role": "user", "content": condensation_prompt}], 100)
            return _extract_response_text(response)
        except Exception:
            return prompt

    retrieved_chunks: List[str] = []
    citations: List[str] = []

    if ensure_vector_index(VECTOR_STORE_DIR, VECTOR_COLLECTION_NAME):
        try:
            # --- STEP A: RETRIEVAL ---
            # 1. Rewrite the query so the database can understand pronouns and context
            search_query = condense_query(user_prompt, history_messages)
            
            _debug_log(f"Searching vector store for condensed query: {search_query}")
            
            # 2. Query the vector store with the CONTEXTUALIZED search query
            retrieved_chunks, citations = query_vector_store(search_query, VECTOR_COLLECTION_NAME, VECTOR_STORE_DIR)
            
            if not retrieved_chunks:
                return (
                    "System Error: No relevant guidance was retrieved from the vector store. "
                    "Please verify the vector index has been created and that the Foundry embedding model is supported.",
                    []
                )
        except Exception as e:
            return (f"System Error: Vector store retrieval failed: {e}", [])
    else:
        return (
            "System Error: The vector index is missing or could not be created. Please run ingestion again.",
            []
        )

    retrieved_chunks = trim_context_chunks(retrieved_chunks)
    context = "\n\n".join(retrieved_chunks)

    prompt_text = (
        "Answer using only the context below. Keep the answer concise and directly relevant. "
        "If the answer cannot be found in the context, reply exactly: I couldn't find a clear answer in approved guidance.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{user_prompt}"
    )

    try:
        response = _call_foundry(build_chat_messages(prompt_text), MAX_FOUNDARY_OUTPUT_TOKENS)
        ai_answer = _extract_response_text(response)

        if "couldn't find a clear answer" in ai_answer.lower() and citations:
            _debug_log("Received refusal from strict prompt; retrying with summarization prompt.")
            summary_prompt = (
                "Use the context to answer the question. If relevant guidance is present, summarize it directly. "
                "Do not reply with a refusal.\n\n"
                f"Context:\n{context}\n\n"
                f"Question:\n{user_prompt}"
            )
            response = _call_foundry(build_chat_messages(summary_prompt), MAX_FOUNDARY_OUTPUT_TOKENS)
            ai_answer = _extract_response_text(response)

        if "couldn't find a clear answer" in ai_answer.lower():
            return "I couldn't find a clear answer in approved guidance.", citations
        return ai_answer, citations

    except Exception as e:
        return f"System Error: {str(e)}", []
