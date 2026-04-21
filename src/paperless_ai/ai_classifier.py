import logging

from django.contrib.auth.models import User

from documents.models import Correspondent
from documents.models import Document
from documents.models import Tag
from documents.permissions import get_objects_for_user_owner_aware
from paperless.config import AIConfig
from paperless_ai.client import AIClient
from paperless_ai.indexing import query_similar_documents
from paperless_ai.indexing import truncate_content

logger = logging.getLogger("paperless_ai.rag_classifier")


def get_default_prompt_template() -> str:
    return """You are a document classification assistant.

Analyze the following document and extract the following information:
- A short descriptive title
- Tags that reflect the content
- Names of people or organizations mentioned
- The type or category of the document
- Suggested folder paths for storing the document
- Up to 3 relevant dates in YYYY-MM-DD format

Filename:
{filename}

Content:
{content}"""


def get_available_tags(user: User | None = None) -> str:
    if user:
        tags = get_objects_for_user_owner_aware(user, "view_tag", Tag)
    else:
        tags = Tag.objects.all()

    tag_names = [tag.name for tag in tags]
    return ", ".join(tag_names) if tag_names else "No tags available"


def get_available_correspondents(user: User | None = None) -> str:
    if user:
        correspondents = get_objects_for_user_owner_aware(
            user,
            "view_correspondent",
            Correspondent,
        )
    else:
        correspondents = Correspondent.objects.all()

    correspondent_names = [c.name for c in correspondents]
    return (
        ", ".join(correspondent_names)
        if correspondent_names
        else "No correspondents available"
    )


def build_prompt_without_rag(
    document: Document,
    user: User | None = None,
) -> str:
    ai_config = AIConfig()
    template = ai_config.llm_prompt_template or get_default_prompt_template()

    filename = document.filename or ""
    content = truncate_content(document.content[:4000] or "")
    available_tags = get_available_tags(user)
    available_correspondents = get_available_correspondents(user)

    try:
        return template.format(
            filename=filename,
            content=content,
            available_tags=available_tags,
            available_correspondents=available_correspondents,
        ).strip()
    except KeyError as e:
        logger.warning(
            "Invalid placeholder in prompt template: %s. Using default template.",
            e,
        )
        return (
            get_default_prompt_template()
            .format(
                filename=filename,
                content=content,
            )
            .strip()
        )


def build_prompt_with_rag(document: Document, user: User | None = None) -> str:
    base_prompt = build_prompt_without_rag(document, user)
    context = truncate_content(get_context_for_document(document, user))

    return f"""{base_prompt}

Additional context from similar documents:
{context}""".strip()


def get_context_for_document(
    doc: Document,
    user: User | None = None,
    max_docs: int = 5,
) -> str:
    visible_documents = (
        get_objects_for_user_owner_aware(
            user,
            "view_document",
            Document,
        )
        if user
        else None
    )
    similar_docs = query_similar_documents(
        document=doc,
        document_ids=[document.pk for document in visible_documents]
        if visible_documents
        else None,
    )[:max_docs]
    context_blocks = []
    for similar in similar_docs:
        text = similar.content[:1000] or ""
        title = similar.title or similar.filename or "Untitled"
        context_blocks.append(f"TITLE: {title}\n{text}")
    return "\n\n".join(context_blocks)


def parse_ai_response(raw: dict) -> dict:
    return {
        "title": raw.get("title", ""),
        "tags": raw.get("tags", []),
        "correspondents": raw.get("correspondents", []),
        "document_types": raw.get("document_types", []),
        "storage_paths": raw.get("storage_paths", []),
        "dates": raw.get("dates", []),
    }


def get_ai_document_classification(
    document: Document,
    user: User | None = None,
) -> dict:
    ai_config = AIConfig()

    prompt = (
        build_prompt_with_rag(document, user)
        if ai_config.llm_embedding_backend
        else build_prompt_without_rag(document, user)
    )

    client = AIClient()
    result = client.run_llm_query(prompt)
    return parse_ai_response(result)
