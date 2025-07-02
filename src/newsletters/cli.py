"""Helpful CLI for newsletters development"""

import typer
import asyncio
from .interview_rag import run_interview_agent, setup_database, create_sample_interviews

app = typer.Typer()


@app.command()
def setup_rag():
    """Set up the RAG database schema for interview insights"""
    asyncio.run(setup_database())
    typer.echo("✅ Database schema created successfully!")


@app.command()
def create_samples():
    """Create sample interview data for testing"""
    asyncio.run(setup_database())
    asyncio.run(create_sample_interviews())
    typer.echo("✅ Sample interviews created!")


@app.command()
def ask(question: str = typer.Argument("What are the main frustrations people are experiencing?")):
    """Ask a question about organizational insights from interviews"""
    typer.echo(f"🤔 Question: {question}")
    typer.echo("🔍 Searching through interview data...")

    result = asyncio.run(run_interview_agent(question))
    typer.echo(f"\n🤖 Answer: {result}")


if __name__ == "__main__":
    app()
