"""
End-to-end RAG test with table image surfacing.

When a query asks for a full table, the response includes a reference
to the rendered table image (in data/extracted_tables/) instead of
relying on broken markdown reconstruction.

For specific row lookups (e.g. "amikacin dose for 30 week neonate"),
the existing row-by-row retrieval is unchanged.
"""

import sys
import json
import os
import time
from datetime import datetime
import requests
import psutil
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from abbreviations import expand_query

console = Console()

CHROMA_DIR = "data/chroma_db"
TABLE_IMAGES_DIR = "data/extracted_tables"
COLLECTION_NAME = "guidelines"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "gemma4:e2b"
OLLAMA_URL = "http://localhost:11434"
TOP_K = 5
SESSION_LOG_DIR = "data/session_logs"

TABLE_REQUEST_KEYWORDS = [
    "table", "list", "schedule", "chart", "all doses", "all values",
    "complete", "full", "entire", "every row", "show me the",
    "give me the", "what are all",
]

SYSTEM_PROMPT = """You are a clinical guideline assistant for hospital staff.

CLINICAL INTERPRETATION:
When interpreting questions, use standard clinical conventions: 
"follow-up" typically refers to post-discharge surveillance and 
appointments, "monitoring" refers to inpatient observation during 
the current admission, and "criteria" usually refers to specific 
thresholds or conditions for a clinical decision.

ANSWERING RULES:
1. Answer using the provided guideline excerpts below.

2. Only refuse with "This isn't covered in the available guidelines. 
   Please consult a senior colleague." if the relevant information is 
   genuinely absent — not just because the wording differs.

3. For specific drug doses, dosing intervals, numeric thresholds, or 
   exact values: only state values that appear verbatim in the provided 
   excerpts. If the specific value the user asks for is not present, 
   refuse rather than estimate, calculate from general knowledge, or 
   extrapolate from related values.

4. Never fabricate drug names, indications, contraindications, or 
   procedures that aren't in the provided excerpts.

5. For calculations:
   (a) only use input values stated in the question or in the excerpts;
   (b) show every step of your working with units at each step;
   (c) if a required input value is missing, state what's missing and 
   refuse to calculate.

6. For patient scenarios involving age: identify whether each value 
   refers to gestational age, postnatal age, corrected age, or 
   treatment duration before answering, and state your interpretation 
   briefly at the start of your response.
   
   Default conventions when the question doesn't specify:
   - "X weeks" or "X-week" alone almost always means gestational age 
     at birth (e.g., "28 weeks" = born at 28 weeks gestation). This 
     is true for any value typically within human gestation range 
     (roughly 20-42 weeks).
   - "X days old" or "day X" means postnatal age (days since birth).
   - "X months old" means postnatal age unless explicitly stated as 
     corrected age.
   - When both gestational and postnatal age are given (e.g., "28 week, 
     day 1"), apply both correctly: gestational age determines patient 
     population (preterm/term), postnatal age determines current 
     management timing.
   
   If after applying these conventions the question still permits two 
   interpretations of age that would lead to different answers (e.g., 
   different drug doses, different management), refuse with: "I'm 
   unsure whether you mean [interpretation A] or [interpretation B], 
   and this affects [the specific clinical decision]. Please rephrase 
   with explicit ages — for example, 'gestational age X weeks, postnatal 
   age Y days'."
   
   For non-dosing questions where interpretation doesn't change the 
   answer, proceed and state the assumption clearly.

7. If the retrieved excerpts contain partial information, state what you 
   can answer from them and explicitly note what additional information 
   would be needed for completeness.

FORMAT RULES:
8. Write the answer as clean prose. Do NOT include source citations in 
   your answer text — sources are displayed separately.

9. If the user asks to see a table, list, regimen, or schedule (signal 
   phrases: "show me the...", "what are all...", "give me the table") 
   AND a [TABLE IMAGE AVAILABLE] marker is present in the context, 
   write a one-sentence summary like "The full dosing table is shown 
   below from [guideline name], page X." Do NOT reproduce the table 
   contents in text.

10. If the user asks for tabular content but no [TABLE IMAGE AVAILABLE] 
    marker is present, answer only with values explicitly in the retrieved 
    excerpts — do not reconstruct missing rows or extrapolate.

11. For specific value lookups (e.g., "amikacin dose for a 30-week baby"), 
    answer directly from the table row data without referring to images.

12. Keep answers concise and clinically actionable.
"""


def get_memory_mb() -> float:
    """Return current process memory usage in MB (resident set size)."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

def get_system_memory() -> dict:
    """Return system-wide memory stats."""
    vm = psutil.virtual_memory()
    return {
        "total_mb": round(vm.total / 1024 / 1024, 0),
        "used_mb": round(vm.used / 1024 / 1024, 0),
        "percent": round(vm.percent, 1),
    }


def query_wants_full_table(query: str) -> bool:
    q_lower = query.lower()
    return any(kw in q_lower for kw in TABLE_REQUEST_KEYWORDS)


def load_collection():
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(name=COLLECTION_NAME, embedding_function=ef)


def retrieve(collection, query: str, top_k: int = TOP_K) -> list[dict]:
    """Retrieve top-k chunks for a query."""
    results = collection.query(query_texts=[query], n_results=top_k)
    
    if not results["documents"] or not results["documents"][0]:
        return []
    
    chunks = []
    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "content": meta.get("display_text", doc),
            "source": meta["source"],
            "page": meta["page"],
            "section": meta.get("section"),
            "chunk_type": meta.get("chunk_type", "text"),
            "chunk_index": meta.get("chunk_index"),
            "table_image": meta.get("table_image"),
            "similarity": round(1.0 - distance, 3),
        })
    return chunks


def detect_table_image_to_show(query: str, chunks: list[dict]) -> dict | None:
    """Decide whether to surface a table image."""
    if not query_wants_full_table(query):
        return None

    for chunk in chunks:
        if chunk.get("table_image"):
            return {
                "filename": chunk["table_image"],
                "path": os.path.join(TABLE_IMAGES_DIR, chunk["table_image"]),
                "source": chunk["source"],
                "page": chunk["page"],
                "section": chunk.get("section"),
            }

    return None


def format_context(chunks: list[dict], table_image_info: dict | None = None) -> str:
    """Build the context block to inject into the prompt."""
    parts = []

    if table_image_info:
        section = (
            f", section '{table_image_info['section']}'"
            if table_image_info.get("section") else ""
        )
        parts.append(
            f"[TABLE IMAGE AVAILABLE - from {table_image_info['source']}, "
            f"page {table_image_info['page']}{section}]\n"
            f"A rendered image of the full table will be displayed to the user "
            f"by the app. You do not need to reproduce its contents in text."
        )

    for chunk in chunks:
        type_label = {
            "text": "TEXT",
            "table_row": "TABLE ROW",
            "table_summary": "TABLE",
        }.get(chunk["chunk_type"], "TEXT")
        
        section = f" — {chunk['section']}" if chunk.get("section") else ""
        header = f"[{type_label} — {chunk['source']}, Page {chunk['page']}{section}]"
        parts.append(f"{header}\n{chunk['content']}")
    
    return "\n\n---\n\n".join(parts)


def generate(question: str, chunks: list[dict],
             table_image_info: dict | None = None) -> tuple[str, dict]:
    """Call Gemma via Ollama's native chat API, streaming the response.
    
    Returns: (answer_text, generation_stats_dict)
    """
    
    context = format_context(chunks, table_image_info=table_image_info)
    user_prompt = f"""GUIDELINE EXCERPTS:

{context}

---

QUESTION: {question}"""

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": True,
        "options": {
            "temperature": 0.1,
            "num_predict": 1024,
        },
    }
    
    full_response = ""
    first_token_time = None
    final_data = {}
    start = time.perf_counter()
    
    with requests.post(
        f"{OLLAMA_URL}/api/chat",
        json=payload,
        stream=True,
        timeout=120,
    ) as response:
        response.raise_for_status()
        
        for line in response.iter_lines():
            if not line:
                continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            if "message" in data and "content" in data["message"]:
                token = data["message"]["content"]
                if token:
                    if first_token_time is None:
                        first_token_time = time.perf_counter() - start
                    console.print(token, end="", style="white")
                    full_response += token
            
            if data.get("done"):
                final_data = data
                break
    
    console.print()
    total_time = time.perf_counter() - start
    
    stats = {
        "ttft_seconds": round(first_token_time, 3) if first_token_time else None,
        "generation_seconds": round(total_time, 3),
        "tokens_generated": final_data.get("eval_count"),
        "tokens_per_second": None,
    }
    
    if final_data.get("eval_count") and final_data.get("eval_duration"):
        eval_secs = final_data["eval_duration"] / 1e9
        stats["tokens_per_second"] = round(final_data["eval_count"] / eval_secs, 1)
    
    return full_response, stats


def check_ollama() -> bool:
    """Verify Ollama is running and model is available."""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        console.print(f"[red]Ollama not reachable at {OLLAMA_URL}[/red]")
        console.print(f"[red]Error: {e}[/red]")
        console.print("\nMake sure Ollama is running:")
        console.print("  ollama serve")
        return False
    
    tags = response.json().get("models", [])
    model_names = [m.get("name", "") for m in tags]
    
    matched = any(LLM_MODEL in name for name in model_names)
    
    if not matched:
        console.print(f"[yellow]Model {LLM_MODEL} not found in Ollama[/yellow]")
        console.print(f"[yellow]Available: {', '.join(model_names) or '(none)'}[/yellow]")
        console.print("\nPull it with:")
        console.print(f"  ollama pull {LLM_MODEL}")
        return False
    
    console.print(f"[green]✓ Ollama connected, {LLM_MODEL} available[/green]")
    return True


def display_metrics(timings: dict, gen_stats: dict, memory: dict):
    """Show a compact metrics table after each query."""
    table = Table(title="Performance", show_header=False, box=None, padding=(0, 2))
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Retrieval", f"{timings['retrieval_ms']:.1f} ms")
    if gen_stats.get("ttft_seconds") is not None:
        table.add_row("Time to first token", f"{gen_stats['ttft_seconds']:.2f} s")
    table.add_row("Generation total", f"{gen_stats['generation_seconds']:.2f} s")
    if gen_stats.get("tokens_per_second"):
        table.add_row(
            "Throughput",
            f"{gen_stats['tokens_per_second']} tok/s "
            f"({gen_stats['tokens_generated']} tokens)"
        )
    table.add_row("Total pipeline", f"{timings['total_seconds']:.2f} s")
    table.add_row("", "")
    table.add_row("Process RAM", f"{memory['process_mb']:.0f} MB")
    table.add_row(
        "System RAM",
        f"{memory['system_used_mb']:.0f} / {memory['system_total_mb']:.0f} MB "
        f"({memory['system_percent']:.1f}%)"
    )
    console.print()
    console.print(table)


def answer_question(question: str, collection) -> dict | None:
    """Full pipeline: expand → retrieve → generate → metrics → return record."""
    pipeline_start = time.perf_counter()
    
    expanded = expand_query(question)
    if expanded != question:
        console.print(f"[dim]Expanded: {expanded}[/dim]")
    console.print(Panel(question, title="Question", border_style="cyan"))
    
    console.print("\n[dim]Retrieving relevant chunks...[/dim]")
    retrieval_start = time.perf_counter()
    chunks = retrieve(collection, expanded)

    table_image_info = detect_table_image_to_show(expanded, chunks)

    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
    
    if not chunks:
        console.print("[red]No relevant chunks found[/red]")
        return None
    
    console.print("\n[bold]Retrieved sources:[/bold]")
    for i, chunk in enumerate(chunks, 1):
        section = f" — {chunk['section']}" if chunk.get("section") else ""
        console.print(
            f"  {i}. [cyan]{chunk['source']}[/cyan] p.{chunk['page']}{section} "
            f"[dim]({chunk['chunk_type']}, sim={chunk['similarity']:.2f})[/dim]"
        )

    if table_image_info:
        console.print(
            f"\n[green]Table image will be displayed: "
            f"{table_image_info['filename']}[/green]"
        )
    
    console.print("\n[bold]Answer:[/bold]\n")
    answer, gen_stats = generate(expanded, chunks, table_image_info=table_image_info)

    if table_image_info:
        console.print(
            f"\n[dim italic]-> Table image: {table_image_info['path']}[/dim italic]"
        )
    
    total_seconds = time.perf_counter() - pipeline_start
    timings = {
        "retrieval_ms": round(retrieval_ms, 1),
        "total_seconds": round(total_seconds, 3),
    }
    
    sys_mem = get_system_memory()
    memory = {
        "process_mb": round(get_memory_mb(), 0),
        "system_used_mb": sys_mem["used_mb"],
        "system_total_mb": sys_mem["total_mb"],
        "system_percent": sys_mem["percent"],
    }
    
    display_metrics(timings, gen_stats, memory)
    
    # Build structured Q&A record
    record = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "expanded_query": expanded if expanded != question else None,
        "answer": answer.strip(),
        "sources": [
            {
                "guideline": chunk["source"],
                "page": chunk["page"],
                "section": chunk.get("section"),
                "chunk_type": chunk["chunk_type"],
                "similarity": chunk["similarity"],
                "excerpt": chunk["content"][:300],
                "table_image": chunk.get("table_image"),
            }
            for chunk in chunks
        ],
        "table_image_displayed": table_image_info,
        "model": LLM_MODEL,
        "performance": {**timings, **gen_stats, **memory},
    }
    
    return record


def save_session(qa_pairs: list[dict]):
    """Save all QA pairs from this session to a timestamped JSON file."""
    if not qa_pairs:
        return
    os.makedirs(SESSION_LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SESSION_LOG_DIR, f"session_{timestamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=2)
    console.print(f"\n[green]Session saved → {path} ({len(qa_pairs)} QA pairs)[/green]")


def main():
    try:
        collection = load_collection()
    except Exception as e:
        console.print(f"[red]Failed to load ChromaDB: {e}[/red]")
        console.print("Run ingest.py first.")
        return
    
    if not check_ollama():
        return
    
    sys_mem = get_system_memory()
    console.print(
        f"[dim]Baseline: process {get_memory_mb():.0f} MB, "
        f"system {sys_mem['percent']:.1f}% used "
        f"({sys_mem['used_mb']:.0f}/{sys_mem['total_mb']:.0f} MB)[/dim]"
    )
    
    qa_pairs = []

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        record = answer_question(question, collection)
        if record:
            qa_pairs.append(record)
        save_session(qa_pairs)
        return
    
    console.print(Panel(
        "[bold]Interactive RAG Testing[/bold]\n"
        "[dim]Ask questions to see retrieval + generation in action.\n"
        "Type 'quit' to exit.[/dim]",
        border_style="blue",
    ))
    
    try:
        while True:
            try:
                question = Prompt.ask("\n[bold green]Question[/bold green]")
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Goodbye![/dim]")
                break
            
            if not question.strip():
                continue
            if question.lower() in ("quit", "exit", "q"):
                break
            
            record = answer_question(question, collection)
            if record:
                qa_pairs.append(record)
    finally:
        save_session(qa_pairs)


if __name__ == "__main__":
    main()