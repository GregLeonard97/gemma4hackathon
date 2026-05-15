"""
Interactive retrieval testing.

Run from project root:
    python workstation/ingestion/query.py

Loads the existing ChromaDB and lets you type questions to see what
chunks get retrieved. Useful for iterating on chunking strategy and
system prompts without re-running ingestion.

Press Ctrl+C or type 'quit' to exit.
"""

import os
import json
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()

# --- Config (same as ingest.py) ---
CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "guidelines"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_TOP_K = 5


def load_collection():
    """Load the existing ChromaDB collection."""
    if not os.path.exists(CHROMA_DIR):
        console.print(f"[red]No database found at {CHROMA_DIR}[/red]")
        console.print("Run ingest.py first to create the database.")
        return None

    ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
        )
    except Exception as e:
        console.print(f"[red]Failed to load collection: {e}[/red]")
        return None

    return collection


def print_result(doc: str, meta: dict, distance: float, rank: int):
    """Pretty-print a single retrieval result."""
    chunk_type = meta.get("chunk_type", "text")
    icon = "📊" if chunk_type == "table" else "📄"
    similarity = 1.0 - distance  # cosine distance → similarity

    header = (
        f"[bold]#{rank}[/bold]  {icon} "
        f"[cyan]{meta['source']}[/cyan] "
        f"[dim]page {meta['page']}[/dim] "
        f"[yellow]similarity: {similarity:.3f}[/yellow]"
    )

    # Truncate long chunks for readability
    display_text = doc if len(doc) <= 500 else doc[:500] + "..."

    panel = Panel(
        display_text,
        title=header,
        title_align="left",
        border_style="dim",
    )
    console.print(panel)

    # If it's a table, also show the structured data in a compact form
    if chunk_type == "table" and "table_data" in meta:
        try:
            table_data = json.loads(meta["table_data"])
            if len(table_data) > 1:
                console.print(
                    f"  [dim]Table has {len(table_data)} rows, "
                    f"{len(table_data[0])} columns[/dim]"
                )
        except (json.JSONDecodeError, KeyError, TypeError):
            pass


def run_query(collection, query: str, top_k: int):
    """Run a single query and display results."""
    console.print(f"\n[bold]Query:[/bold] {query}")
    console.print(f"[dim]Retrieving top {top_k} chunks...[/dim]\n")

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
    )

    if not results["documents"] or not results["documents"][0]:
        console.print("[red]No results found[/red]")
        return

    for rank, (doc, meta, distance) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ),
        start=1,
    ):
        print_result(doc, meta, distance, rank)


def show_stats(collection):
    """Print summary stats about the collection."""
    total = collection.count()

    if total == 0:
        console.print("[yellow]Collection is empty — run ingest.py first[/yellow]")
        return

    # Sample to get chunk type breakdown
    sample = collection.get(limit=min(total, 5000))
    type_counts = {}
    sources = set()

    for meta in sample["metadatas"]:
        ctype = meta.get("chunk_type", "text")
        type_counts[ctype] = type_counts.get(ctype, 0) + 1
        sources.add(meta.get("source", "unknown"))

    console.print(Panel(
        f"Total chunks: [bold]{total}[/bold]\n"
        f"Unique sources: [bold]{len(sources)}[/bold]\n"
        f"Type breakdown: {type_counts}",
        title="Collection Stats",
        border_style="green",
    ))


def main():
    console.print(
        Panel(
            "[bold]Guideline Retrieval Tester[/bold]\n"
            "[dim]Type a question to see what chunks get retrieved.\n"
            "Commands: 'stats' for collection info, 'quit' to exit.\n"
            "Prefix with '@N ' to change top_k, e.g. '@3 sepsis protocol'[/dim]",
            border_style="blue",
        )
    )

    collection = load_collection()
    if collection is None:
        return

    show_stats(collection)

    while True:
        try:
            query = Prompt.ask("\n[bold green]Query[/bold green]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        query = query.strip()

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            break

        if query.lower() == "stats":
            show_stats(collection)
            continue

        # Parse optional @N prefix for custom top_k
        top_k = DEFAULT_TOP_K
        if query.startswith("@"):
            parts = query.split(" ", 1)
            if len(parts) == 2:
                try:
                    top_k = int(parts[0][1:])
                    query = parts[1]
                except ValueError:
                    pass

        run_query(collection, query, top_k)


if __name__ == "__main__":
    main()