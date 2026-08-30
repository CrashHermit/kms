import asyncio
from pathlib import Path

from kms.runtime import Runtime


async def main() -> None:
    output_dir = Path('/tmp/kms-training-dataset')
    output_dir.mkdir(parents=True, exist_ok=True)
    async with Runtime() as application:
        await application.ingest(
            '/tmp/kms-history/ten-great-events.pdf',
            output_dir=output_dir,
            ocr_response_path=(
                '/tmp/kms-history/ten-events-ingest-refined-2/'
                'ocr_response.json'
            ),
            source='training-dataset',
        )


asyncio.run(main())
