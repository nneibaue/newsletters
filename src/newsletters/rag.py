"""
RAG (Retrieval-Augmented Generation) Agent for Documentation Q&A

This module implements a retrieval-augmented generation (RAG) agent using OpenAI's GPT models and a PostgreSQL vector database. It enables question answering over documentation by embedding queries and documentation sections, retrieving the most relevant sections, and generating answers using an LLM. The module also provides utilities to build and manage the search database from a remote JSON document, and includes tools for embedding, retrieval, and database management.
"""

# =========================
# Module Overview
# =========================
# This file implements a RAG (Retrieval-Augmented Generation) agent for answering questions over documentation.
# It uses OpenAI's GPT models for LLM and embedding, asyncpg for PostgreSQL vector search, and logfire for logging.
#
# Key features:
# - Embeds documentation sections and stores them in a vector database (PostgreSQL + pgvector)
# - Embeds user queries and retrieves the most relevant documentation sections
# - Uses an LLM to answer questions using retrieved context
# - Includes utilities to build the search database from a remote JSON file
# - CLI entrypoints for building the DB and running a search


from __future__ import annotations as _annotations

import asyncio
import re
import sys
import unicodedata
from contextlib import asynccontextmanager
from dataclasses import dataclass

import asyncpg # type: ignore[import-untyped]
import httpx
import logfire
import pydantic_core
from openai import AsyncOpenAI
from pydantic import TypeAdapter
from pydantic_ai import Agent
from typing_extensions import AsyncGenerator

from pydantic_ai import RunContext

# Configure logfire for logging and instrumentation
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_asyncpg()
logfire.instrument_pydantic_ai()


# =========================
# Dependency Container
# =========================
@dataclass
class Deps:
    openai: AsyncOpenAI  # OpenAI async client for embeddings and LLM
    pool: asyncpg.Pool   # PostgreSQL connection pool

agent = Agent('openai:gpt-4o', deps_type=Deps)

# =========================
# Agent and Retrieval Tool
# =========================
# Create an agent using OpenAI GPT-4o and the Deps dataclass for dependency injection

@agent.tool
async def retrieve(context: RunContext[Deps], search_query: str) -> str:
    """
    Retrieve documentation sections based on a search query.
    - Embeds the search query using OpenAI's embedding model
    - Queries the database for the most similar documentation sections using vector similarity
    - Returns the top 8 matching sections formatted as Markdown
    """
    with logfire.span(
        'create embedding for {search_query=}', search_query=search_query
    ):
        embedding = await context.deps.openai.embeddings.create(
            input=search_query,
            model='text-embedding-3-small',
        )

    assert len(embedding.data) == 1, (
        f'Expected 1 embedding, got {len(embedding.data)}, doc query: {search_query!r}'
    )
    embeddingg = embedding.data[0].embedding
    embedding_json = pydantic_core.to_json(embeddingg).decode()
    rows = await context.deps.pool.fetch(
        'SELECT url, title, content FROM doc_sections ORDER BY embedding <-> $1 LIMIT 8',
        embedding_json,
    )
    return '\n\n'.join(
        f'# {row["title"]}\nDocumentation URL:{row["url"]}\n\n{row["content"]}\n'
        for row in rows
    )


# =========================
# Agent Entrypoint
# =========================
async def run_agent(question: str):
    """
    Entry point to run the agent and perform RAG based question answering.
    - Sets up OpenAI client and logging
    - Connects to the database
    - Runs the agent with the provided question and prints the answer
    """
    openai = AsyncOpenAI()
    logfire.instrument_openai(openai)

    logfire.info('Asking "{question}"', question=question)


    async with database_connect(False) as pool:
        deps = Deps(openai=openai, pool=pool)
        answer = await agent.run(question, deps=deps)
    print(answer.output)


#######################################################
# The rest of this file is dedicated to preparing the #
# search database, and some utilities.                #
#######################################################

# =========================
# Documentation Source
# =========================
# URL to the JSON document containing documentation sections
DOCS_JSON = (
    'https://gist.githubusercontent.com/'
    'samuelcolvin/4b5bb9bb163b1122ff17e29e48c10992/raw/'
    '80c5925c42f1442c24963aaf5eb1a324d47afe95/logfire_docs.json'
)

# =========================
# Build Search Database
# =========================
async def build_search_db():
    """
    Build the search database.
    - Downloads documentation sections from DOCS_JSON
    - Validates and parses them
    - Embeds each section and inserts it into the database with vector index
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(DOCS_JSON)
        response.raise_for_status()
    sections = sessions_ta.validate_json(response.content)

    openai = AsyncOpenAI()
    logfire.instrument_openai(openai)

    async with database_connect(True) as pool:
        with logfire.span('create schema'):
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(DB_SCHEMA)

        sem = asyncio.Semaphore(10)  # Limit concurrent embedding requests
        async with asyncio.TaskGroup() as tg:
            for section in sections:
                tg.create_task(insert_doc_section(sem, openai, pool, section))


# =========================
# Insert Documentation Section
# =========================
async def insert_doc_section(
    sem: asyncio.Semaphore,
    openai: AsyncOpenAI,
    pool: asyncpg.Pool,
    section: DocsSection,
) -> None:
    """
    Insert a documentation section into the database with its embedding.
    - Checks if the section already exists by URL
    - Embeds the section content
    - Inserts into the doc_sections table
    """
    async with sem:
        url = section.url()
        exists = await pool.fetchval('SELECT 1 FROM doc_sections WHERE url = $1', url)
        if exists:
            logfire.info('Skipping {url=}', url=url)
            return

        with logfire.span('create embedding for {url=}', url=url):
            embedding = await openai.embeddings.create(
                input=section.embedding_content(),
                model='text-embedding-3-small',
            )
        assert len(embedding.data) == 1, (
            f'Expected 1 embedding, got {len(embedding.data)}, doc section: {section}'
        )
        embedding = embedding.data[0].embedding # type: ignore
        embedding_json = pydantic_core.to_json(embedding).decode()
        await pool.execute(
            'INSERT INTO doc_sections (url, title, content, embedding) VALUES ($1, $2, $3, $4)',
            url,
            section.title,
            section.content,
            embedding_json,
        )


# =========================
# DocsSection Dataclass
# =========================
@dataclass
class DocsSection:
    id: int
    parent: int | None
    path: str
    level: int
    title: str
    content: str

    def url(self) -> str:
        """
        Generate a documentation URL for the section.
        """
        url_path = re.sub(r'\.md$', '', self.path)
        return (
            f'https://logfire.pydantic.dev/docs/{url_path}/#{slugify(self.title, "-")}'
        )

    def embedding_content(self) -> str:
        """
        Content to use for embedding (path, title, and content).
        """
        return '\n\n'.join((f'path: {self.path}', f'title: {self.title}', self.content))


# TypeAdapter for validating/parsing the list of DocsSection
sessions_ta = TypeAdapter(list[DocsSection])


# =========================
# Database Connection Utility
# =========================
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
@asynccontextmanager
async def database_connect(
    create_db: bool = False,
) -> AsyncGenerator[asyncpg.Pool, None]:
    """
    Connect to the PostgreSQL database, optionally creating it if needed.
    Yields an asyncpg connection pool.
    """
    server_dsn, database = (
        'postgresql://postgres:postgres@host.docker.internal:54320',
        'pydantic_ai_rag',
    )
    if create_db:
        with logfire.span('check and create DB'):
            conn = await asyncpg.connect(server_dsn)
            try:
                db_exists = await conn.fetchval(
                    'SELECT 1 FROM pg_database WHERE datname = $1', database
                )
                if not db_exists:
                    await conn.execute(f'CREATE DATABASE {database}')
            finally:
                await conn.close()

    pool = await asyncpg.create_pool(f'{server_dsn}/{database}')
    try:
        yield pool
    finally:
        await pool.close()


# =========================
# Database Schema
# =========================
DB_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS doc_sections (
    id serial PRIMARY KEY,
    url text NOT NULL UNIQUE,
    title text NOT NULL,
    content text NOT NULL,
    -- text-embedding-3-small returns a vector of 1536 floats
    embedding vector(1536) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_doc_sections_embedding ON doc_sections USING hnsw (embedding vector_l2_ops);
"""


# =========================
# Slugify Utility
# =========================
def slugify(value: str, separator: str, unicode: bool = False) -> str:
    """
    Slugify a string, to make it URL friendly.
    """
    # Taken unchanged from https://github.com/Python-Markdown/markdown/blob/3.7/markdown/extensions/toc.py#L38
    if not unicode:
        # Replace Extended Latin characters with ASCII, i.e. `žlutý` => `zluty`
        value = unicodedata.normalize('NFKD', value)
        value = value.encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    return re.sub(rf'[{separator}\s]+', separator, value)


# =========================
# CLI Entrypoint
# =========================
if __name__ == '__main__':
    action = sys.argv[1] if len(sys.argv) > 1 else None
    if action == 'build':
        asyncio.run(build_search_db())
    elif action == 'search':
        if len(sys.argv) == 3:
            q = sys.argv[2]
        else:
            q = 'How do I configure logfire to work with FastAPI?'
        asyncio.run(run_agent(q))
    else:
        print(
            'uv run --extra examples -m pydantic_ai_examples.rag build|search',
            file=sys.stderr,
        )
        sys.exit(1)
