"""
Semantic chunking utility for parsing structured data
"""
import re
from typing import List
from langchain.schema import Document


class SemanticChunker:
    """Parse optimized data into semantic chunks with rich metadata"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.documents = []
    
    def load_and_parse(self) -> List[Document]:
        """Parse the structured data into semantic chunks"""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by department headers
        departments = re.split(r'=== DEPARTMENT \d+ ===', content)
        departments = [d.strip() for d in departments if d.strip()]
        
        for dept_content in departments:
            self._parse_department(dept_content)
        
        print(f"✅ Parsed {len(self.documents)} semantic chunks")
        return self.documents
    
    def _parse_department(self, dept_content: str):
        """Parse a single department into separate section chunks"""
        lines = dept_content.split('\n')
        
        # Extract department metadata
        dept_name = None
        dept_number = None
        
        for line in lines[:5]:
            line = line.strip()
            if line.startswith('Department Name:'):
                dept_name = line.replace('Department Name:', '').strip()
            elif line.startswith('Department Number:'):
                dept_number = line.replace('Department Number:', '').strip()
        
        if not dept_name:
            return
        
        # Split by section markers
        parts = re.split(r'--- SECTION: (.+?) ---', dept_content)
        section_names = parts[1::2]
        section_contents = parts[2::2]
        
        # Create one document per section
        for section_name, section_content in zip(section_names, section_contents):
            section_name = section_name.strip()
            section_content = section_content.strip()
            
            if not section_content:
                continue
            
            # Create metadata
            metadata = {
                'department_number': dept_number,
                'department_name': dept_name,
                'section': section_name,
                'full_context': f"{dept_name} - {section_name}",
                'source': 'university_prospectus'
            }
            
            # Add section type for filtering
            if 'Introduction' in section_name:
                metadata['section_type'] = 'introduction'
            elif 'Program' in section_name:
                metadata['section_type'] = 'programs'
            elif 'Eligibility' in section_name:
                metadata['section_type'] = 'eligibility'
            elif 'Faculty' in section_name:
                metadata['section_type'] = 'faculty'
            
            doc = Document(
                page_content=section_content,
                metadata=metadata
            )
            
            self.documents.append(doc)
