
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

# Use the built-in granite-docling-258M model locally (no remote VLM server)
pipeline_options = PdfPipelineOptions(
    do_ocr=True,
    do_table_structure=True,
    do_formula_enrichment=True,
    images_scale=3.0,  # Higher resolution for better OCR/formula recognition
)

# 3. Layout/OCR handles text and structure; built-in model enriches formulas to LaTeX
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

# 4. Convert your document
result = converter.convert("test.pdf")

# 5. Export utilizing your specific documentation constraints
markdown_output = result.document.export_to_markdown(
    traverse_pictures=True,      # Crucial for deep image-based parsing layouts
    compact_tables=False,         # Retains column padding for clearer terminal viewing
    enable_chart_tables=True     # Allows your 4B model to translate charts directly into tables
)

output_file_path = "output_document.md"
with open(output_file_path, "w", encoding="utf-8") as f:
    f.write(markdown_output)

print(f"Successfully saved parsed document to: {output_file_path}")

