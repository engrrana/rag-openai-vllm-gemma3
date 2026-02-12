"""Quick diagnostic script"""
from app.utils.chunker import SemanticChunker
from app.core.config import settings

chunker = SemanticChunker(settings.data_file)
docs = chunker.load_and_parse()

print(f"Total documents: {len(docs)}")
print(f"\nFirst document:")
print(f"  Department: {docs[0].metadata['department_name']}")
print(f"  Section: {docs[0].metadata['section']}")
print(f"  Section Type: {docs[0].metadata.get('section_type', 'N/A')}")
print(f"  Content preview: {docs[0].page_content[:100]}...")
