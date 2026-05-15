"""
Ingest with table-aware chunking AND table image extraction — v5.

Key change from v4: each detected table is also rendered as a PNG image
of just the table region. This lets the app display the actual visual
table for "show me the table" queries while preserving row-by-row
chunking for specific value lookups.

Run from project root:
    python workstation/ingestion/ingest.py
"""

import os
import re
import json
import fitz
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.console import Console
from rich.progress import track

console = Console()

GUIDELINES_DIR = "data/guidelines"
CHROMA_DIR = "data/chroma_db"
IMAGES_DIR = "data/extracted_images"
TABLE_IMAGES_DIR = "data/extracted_tables"
COLLECTION_NAME = "guidelines"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MIN_ROW_CHARS = 15

TABLE_IMAGE_SCALE = 2.0
TABLE_PADDING_PX = 10


def clean_filename_for_display(filename: str) -> str:
    name = os.path.splitext(filename)[0]
    name = re.sub(r'_v\d+', '', name)
    name = re.sub(r'[_\-]', ' ', name)
    return name.strip()


def detect_headings(page: fitz.Page) -> list[tuple[float, str]]:
    blocks = page.get_text("dict")["blocks"]
    
    font_sizes = []
    for block in blocks:
        if block["type"] != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if span.get("text", "").strip():
                    font_sizes.append(span["size"])
    
    if not font_sizes:
        return []
    
    sizes_sorted = sorted(font_sizes)
    body_size = sizes_sorted[len(sizes_sorted) // 2]
    definite_heading_size = body_size * 1.15
    possible_heading_size = body_size * 1.02
    
    headings = []

    for block in blocks:
        if block["type"] != 0:
            continue
        for line in block.get("lines", []):
            line_text = ""
            y_pos = None
            max_size = 0
            is_bold = False
            is_italic = False
            all_spans_formatted = True
            has_any_text = False
            
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if not text:
                    continue
                has_any_text = True
                line_text += span.get("text", "")
                if y_pos is None:
                    y_pos = span["bbox"][1]
                max_size = max(max_size, span["size"])

                flags = span.get("flags", 0)
                font_name = span.get("font", "").lower()
                span_is_bold = bool(flags & 16) or "bold" in font_name or "black" in font_name
                span_is_italic = bool(flags & 2) or "italic" in font_name or "oblique" in font_name

                if span_is_bold:
                    is_bold = True
                if span_is_italic:
                    is_italic = True
                if not (span_is_bold or span_is_italic):
                    all_spans_formatted = False
            
            if not has_any_text:
                continue
            line_text = line_text.strip()
            if not line_text or y_pos is None:
                continue
            if len(line_text) > 120 or len(line_text) < 3:
                continue
            if line_text.endswith('.') and len(line_text) > 50:
                continue

            is_heading = False
            if max_size >= definite_heading_size:
                is_heading = True
            elif max_size >= possible_heading_size and (is_bold or is_italic):
                is_heading = True
            elif all_spans_formatted and (is_bold or is_italic) and len(line_text) < 80:
                if not line_text.endswith((',', ';', ')', '"', "'")):
                    is_heading = True

            if not is_heading:
                continue
            
            headings.append((y_pos, line_text))
    
    return sorted(headings, key=lambda x: x[0])


def find_heading_for_position(headings, y_pos):
    best = None
    for heading_y, heading_text in headings:
        if heading_y <= y_pos:
            best = heading_text
        else:
            break
    return best


def render_table_image(page: fitz.Page, table_bbox, source: str,
                       page_num: int, table_idx: int) -> str | None:
    """Render a table region as a PNG image. Returns filename or None."""
    try:
        rect = fitz.Rect(table_bbox)
        rect.x0 = max(0, rect.x0 - TABLE_PADDING_PX)
        rect.y0 = max(0, rect.y0 - TABLE_PADDING_PX)
        rect.x1 = min(page.rect.width, rect.x1 + TABLE_PADDING_PX)
        rect.y1 = min(page.rect.height, rect.y1 + TABLE_PADDING_PX)

        matrix = fitz.Matrix(TABLE_IMAGE_SCALE, TABLE_IMAGE_SCALE)
        pixmap = page.get_pixmap(matrix=matrix, clip=rect, alpha=False)

        safe_source = re.sub(r'[^\w\-.]', '_', source)
        image_filename = f"{safe_source}_p{page_num}_table{table_idx}.png"
        image_path = os.path.join(TABLE_IMAGES_DIR, image_filename)

        os.makedirs(TABLE_IMAGES_DIR, exist_ok=True)
        pixmap.save(image_path)

        return image_filename
    except Exception as e:
        console.print(f"[yellow]Failed to render table image: {e}[/yellow]")
        return None


def row_to_natural_language(headers: list[str], row: list[str]) -> str:
    if not row or not headers:
        return ""
    
    subject = row[0].strip() if row[0] else ""
    
    pairs = []
    for h, v in zip(headers[1:], row[1:]):
        h = h.strip()
        v = v.strip()
        if h and v:
            pairs.append(f"{h}: {v}")
    
    if not subject and not pairs:
        return ""
    
    if subject and pairs:
        return f"{subject} — {'; '.join(pairs)}"
    elif subject:
        return subject
    else:
        return "; ".join(pairs)


def extract_table_rows(page: fitz.Page, source: str, page_num: int,
                       headings: list) -> list[dict]:
    chunks = []

    try:
        tables = page.find_tables()
    except Exception:
        return chunks

    for table_idx, table in enumerate(tables.tables):
        data = table.extract()
        if not data or len(data) < 2:
            continue

        clean_data = []
        for row in data:
            clean_row = [(cell or "").strip() for cell in row]
            if any(clean_row):
                clean_data.append(clean_row)

        if len(clean_data) < 2:
            continue

        headers = clean_data[0]
        data_rows = clean_data[1:]

        table_y = table.bbox[1]
        section = find_heading_for_position(headings, table_y)

        full_table_json = json.dumps(clean_data)

        # Render table region as image
        table_image_filename = render_table_image(
            page, table.bbox, source, page_num, table_idx
        )
        
        for row_idx, row in enumerate(data_rows):
            row_text = row_to_natural_language(headers, row)
            
            if len(row_text) < MIN_ROW_CHARS:
                continue
            
            full_row_text = f"Table columns: {' | '.join(headers)}\nRow: {row_text}"

            chunk_data = {
                "text": full_row_text,
                "display_text": row_text,
                "source": source,
                "page": page_num,
                "section": section,
                "chunk_type": "table_row",
                "chunk_index": table_idx * 1000 + row_idx,
                "table_headers": json.dumps(headers),
                "table_row_data": json.dumps(row),
                "full_table_data": full_table_json,
            }

            if table_image_filename:
                chunk_data["table_image"] = table_image_filename

            chunks.append(chunk_data)
        
        summary_lines = [f"Table: {' | '.join(headers)}"]
        for row in data_rows[:10]:
            summary_lines.append(row_to_natural_language(headers, row))
        
        summary_text = "\n".join(summary_lines)
        if len(summary_text) >= 50:
            summary_chunk = {
                "text": summary_text,
                "display_text": summary_text,
                "source": source,
                "page": page_num,
                "section": section,
                "chunk_type": "table_summary",
                "chunk_index": table_idx * 1000 + 999,
                "full_table_data": full_table_json,
            }

            if table_image_filename:
                summary_chunk["table_image"] = table_image_filename

            chunks.append(summary_chunk)

    return chunks


def extract_images(page: fitz.Page, source: str, page_num: int) -> list[dict]:
    chunks = []
    image_list = page.get_images(full=True)

    for img_index, img_info in enumerate(image_list):
        xref = img_info[0]
        try:
            base_image = page.parent.extract_image(xref)
        except Exception:
            continue

        if not base_image:
            continue

        image_bytes = base_image["image"]
        image_ext = base_image.get("ext", "png")
        width = base_image.get("width", 0)
        height = base_image.get("height", 0)
        if width < 100 or height < 100:
            continue

        safe_source = re.sub(r'[^\w\-.]', '_', source)
        image_filename = f"{safe_source}_p{page_num}_img{img_index}.{image_ext}"
        image_path = os.path.join(IMAGES_DIR, image_filename)

        os.makedirs(IMAGES_DIR, exist_ok=True)
        with open(image_path, "wb") as f:
            f.write(image_bytes)

        chunks.append({
            "text": f"[Image from {source}, Page {page_num}.]",
            "display_text": f"Image on page {page_num}",
            "source": source,
            "page": page_num,
            "chunk_type": "image",
            "chunk_index": img_index,
            "image_path": image_path,
            "image_filename": image_filename,
        })

    return chunks


def extract_text_chunks(page, source, page_num, table_rects, headings):
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

    text_regions = []
    for block in blocks:
        if block["type"] != 0:
            continue
        block_rect = fitz.Rect(block["bbox"])
        if any(block_rect.intersects(tr) for tr in table_rects):
            continue
        
        block_text = ""
        for line in block.get("lines", []):
            line_text = ""
            for span in line.get("spans", []):
                line_text += span.get("text", "")
            if line_text.strip():
                block_text += line_text.strip() + "\n"
        
        if block_text.strip():
            text_regions.append({"text": block_text.strip(), "y": block_rect.y0})

    if not text_regions:
        return []

    ordered_regions = sorted(text_regions, key=lambda r: r["y"])

    page_regions = []
    for region in ordered_regions:
        region_text = re.sub(r'\s+', ' ', region["text"]).strip()
        if not region_text:
            continue
        page_regions.append({
            "text": region_text,
            "section": find_heading_for_position(headings, region["y"]),
        })

    if not page_regions:
        return []

    page_parts = []
    region_spans = []
    cursor = 0
    for idx, region in enumerate(page_regions):
        if idx > 0:
            separator = "\n\n"
            page_parts.append(separator)
            cursor += len(separator)

        start = cursor
        page_parts.append(region["text"])
        cursor += len(region["text"])
        region_spans.append((start, cursor, region["section"]))

    page_text = "".join(page_parts)
    if not page_text:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    chunk_idx = 0
    search_from = 0

    for piece in splitter.split_text(page_text):
        piece = piece.strip()
        if len(piece) < 40:
            continue

        start_pos = page_text.find(piece, search_from)
        if start_pos == -1:
            start_pos = page_text.find(piece)
        if start_pos == -1:
            start_pos = search_from

        section = None
        for span_start, span_end, span_section in region_spans:
            if span_start <= start_pos < span_end:
                section = span_section
                break

        chunks.append({
            "text": piece,
            "display_text": piece,
            "source": source,
            "page": page_num,
            "section": section,
            "chunk_type": "text",
            "chunk_index": chunk_idx,
        })
        chunk_idx += 1
        search_from = max(search_from + 1, start_pos + 1)

    return chunks


def build_enriched_text(chunk: dict) -> str:
    display_name = clean_filename_for_display(chunk["source"])
    section = chunk.get("section")
    chunk_type = chunk.get("chunk_type", "text")
    
    parts = [f"Document: {display_name}"]
    if section:
        parts.append(f"Section: {section}")
    
    if chunk_type == "table_row":
        parts.append("Type: table row")
    elif chunk_type == "table_summary":
        parts.append("Type: summary table")
    
    parts.append(f"Content: {chunk['text']}")
    return "\n".join(parts)


def process_pdf(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    source = os.path.basename(pdf_path)
    all_chunks = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        headings = detect_headings(page)

        table_chunks = extract_table_rows(page, source, page_num + 1, headings)
        all_chunks.extend(table_chunks)

        table_rects = []
        try:
            tables = page.find_tables()
            for table in tables.tables:
                table_rects.append(fitz.Rect(table.bbox))
        except Exception:
            pass

        text_chunks = extract_text_chunks(page, source, page_num + 1,
                                          table_rects, headings)
        all_chunks.extend(text_chunks)

        image_chunks = extract_images(page, source, page_num + 1)
        all_chunks.extend(image_chunks)

    doc.close()
    return all_chunks


def main():
    console.print("[bold green]Guideline Ingestion Pipeline v5[/bold green]")
    console.print("[dim]Table-aware chunking + table image extraction[/dim]\n")

    if not os.path.exists(GUIDELINES_DIR):
        console.print(f"[red]Directory not found: {GUIDELINES_DIR}[/red]")
        return

    pdf_files = sorted([
        f for f in os.listdir(GUIDELINES_DIR)
        if f.lower().endswith(".pdf")
    ])

    if not pdf_files:
        console.print(f"[red]No PDFs found in {GUIDELINES_DIR}/[/red]")
        return

    console.print(f"Found {len(pdf_files)} PDFs")

    all_chunks = []
    stats = {"text": 0, "table_row": 0, "table_summary": 0, "image": 0}
    table_images_rendered = set()

    for pdf_file in track(pdf_files, description="Processing..."):
        pdf_path = os.path.join(GUIDELINES_DIR, pdf_file)
        chunks = process_pdf(pdf_path)
        for c in chunks:
            stats[c["chunk_type"]] = stats.get(c["chunk_type"], 0) + 1
            if c.get("table_image"):
                table_images_rendered.add(c["table_image"])
        all_chunks.extend(chunks)

    console.print("\nExtracted chunks:")
    for ctype, count in stats.items():
        console.print(f"  {ctype}: {count}")
    console.print(f"  Table images rendered: {len(table_images_rendered)}")

    console.print(f"\nLoading embedding model: {EMBEDDING_MODEL}")

    ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    embeddable_types = ("text", "table_row", "table_summary")
    embeddable = [c for c in all_chunks if c["chunk_type"] in embeddable_types]

    BATCH_SIZE = 200
    for i in track(range(0, len(embeddable), BATCH_SIZE), description="Embedding..."):
        batch = embeddable[i:i + BATCH_SIZE]
        enriched_texts = [build_enriched_text(c) for c in batch]

        metadatas = []
        for c in batch:
            meta = {
                "source": c["source"],
                "page": c["page"],
                "chunk_type": c["chunk_type"],
                "chunk_index": c["chunk_index"],
                "display_text": c.get("display_text", c["text"]),
            }
            if c.get("section"):
                meta["section"] = c["section"]
            if "table_headers" in c:
                meta["table_headers"] = c["table_headers"]
            if "table_row_data" in c:
                meta["table_row_data"] = c["table_row_data"]
            if "full_table_data" in c:
                meta["full_table_data"] = c["full_table_data"]
            if "table_image" in c:
                meta["table_image"] = c["table_image"]
            metadatas.append(meta)

        collection.add(
            documents=enriched_texts,
            metadatas=metadatas,
            ids=[f"chunk_{i+j}" for j in range(len(batch))],
        )

    image_chunks = [c for c in all_chunks if c["chunk_type"] == "image"]
    if image_chunks:
        image_index = [{
            "source": c["source"],
            "page": c["page"],
            "image_filename": c["image_filename"],
            "image_path": c["image_path"],
        } for c in image_chunks]

        index_path = os.path.join(IMAGES_DIR, "image_index.json")
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(image_index, f, indent=2)

    console.print("\n[bold green]Done![/bold green]")
    console.print(f"  {len(embeddable)} chunks embedded")
    console.print(f"  {stats.get('table_row', 0)} table rows searchable")
    console.print(f"  {len(table_images_rendered)} table images rendered to {TABLE_IMAGES_DIR}/")


if __name__ == "__main__":
    main()