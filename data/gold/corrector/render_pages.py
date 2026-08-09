import json
import sys
from pathlib import Path

GOLD = Path(__file__).parent
REPO = GOLD.parents[2]
DEFAULT_OUTPUT = REPO / 'output' / 'gold_pages'


def main() -> None:
    import pypdfium2 as pdfium

    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)

    index = json.loads((GOLD / 'index.json').read_text())
    pages = {
        (record['source_pdf'], record['page'], record['render_scale']): record[
            'page_image'
        ]
        for record in index['records']
    }

    for (pdf_path, page, scale), name in sorted(pages.items()):
        target = output_dir / name
        if target.exists():
            continue
        document = pdfium.PdfDocument(str(REPO / pdf_path))
        try:
            document[page].render(scale=scale).to_pil().save(target)
        finally:
            document.close()
    print(f'{len(pages)} page image(s) in {output_dir}')


if __name__ == '__main__':
    main()

