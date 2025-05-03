import re
import json
import asyncio
from .utils import ChatGPT_API_async, extract_json

################### Financial Table Recognition and Extraction #########################################################

async def detect_financial_tables(page_text, model=None):
    """
    Detect if a page contains financial tables and identify their boundaries.
    
    Args:
        page_text: Text content of the page
        model: LLM model to use
        
    Returns:
        Dictionary with table detection results
    """
    prompt = f"""
    You are a financial document analysis expert. Your task is to detect if the given page contains financial tables.
    
    Page content:
    {page_text}
    
    Please analyze the content and identify if there are any financial tables present. If tables are found, determine:
    1. The type of financial table (balance sheet, income statement, cash flow statement, etc.)
    2. The approximate boundaries (start and end) in the text
    3. Whether the table spans multiple pages (if it appears to be cut off)
    
    Reply format:
    {{
        "tables_detected": true/false,
        "tables": [
            {{
                "table_type": "type of financial table",
                "start_position": "approximate start position in text",
                "end_position": "approximate end position in text",
                "multi_page": true/false,
                "confidence": 0.0-1.0
            }}
        ]
    }}
    Directly return the final JSON structure. Do not output anything else.
    """
    
    response = await ChatGPT_API_async(model=model, prompt=prompt)
    return extract_json(response)

async def extract_table_structure(table_text, table_type, model=None):
    """
    Extract the structure of a financial table including headers, rows, and numerical data.
    
    Args:
        table_text: Text content of the detected table
        table_type: Type of financial table (balance sheet, income statement, etc.)
        model: LLM model to use
        
    Returns:
        Structured representation of the table
    """
    prompt = f"""
    You are a financial table parsing expert. Your task is to extract the structure of a {table_type}.
    
    Table content:
    {table_text}
    
    Please extract:
    1. Column headers
    2. Row labels
    3. Numerical data with proper formatting (preserve parentheses for negative values)
    4. Units of measurement (thousands, millions, billions)
    5. Currency symbols
    6. Footnote references
    
    Reply format:
    {{
        "headers": ["header1", "header2", ...],
        "rows": [
            {{
                "label": "row label",
                "values": ["value1", "value2", ...],
                "footnotes": ["footnote1", "footnote2", ...]
            }}
        ],
        "units": "units of measurement",
        "currency": "currency symbol",
        "footnote_references": ["ref1", "ref2", ...]
    }}
    Directly return the final JSON structure. Do not output anything else.
    """
    
    response = await ChatGPT_API_async(model=model, prompt=prompt)
    return extract_json(response)

async def table_to_json(table_structure, model=None):
    """
    Convert parsed table structure to a standardized JSON format.
    
    Args:
        table_structure: Parsed table structure
        model: LLM model to use
        
    Returns:
        Standardized JSON representation of the table
    """
    # This function converts the extracted table structure to a standardized JSON format
    # based on the table type (balance sheet, income statement, etc.)
    return {
        "table_data": table_structure,
        "metadata": {
            "standardized": True,
            "format_version": "1.0"
        }
    }

################### Regulatory Section Identification #########################################################

async def identify_regulatory_section(section_title, section_content, model=None):
    """
    Identify if a section is a standard regulatory section in SEC filings.
    
    Args:
        section_title: Title of the section
        section_content: Content of the section
        model: LLM model to use
        
    Returns:
        Classification of the regulatory section
    """
    prompt = f"""
    You are an expert in SEC regulatory filings. Your task is to identify if the given section is a standard regulatory section.
    
    Section title: {section_title}
    Section content preview: {section_content[:500]}...
    
    Please analyze the section and determine:
    1. If this is a standard regulatory section required in SEC filings
    2. The exact regulatory section type it corresponds to
    3. Your confidence in this classification
    
    Reply format:
    {{
        "is_regulatory": true/false,
        "section_type": "type of regulatory section",
        "standard_title": "standard title for this section type",
        "confidence": 0.0-1.0,
        "reasoning": "brief explanation for your classification"
    }}
    Directly return the final JSON structure. Do not output anything else.
    """
    
    response = await ChatGPT_API_async(model=model, prompt=prompt)
    return extract_json(response)

async def validate_regulatory_completeness(structure, model=None):
    """
    Validate if all required regulatory sections are present in the document.
    
    Args:
        structure: Document structure with identified sections
        model: LLM model to use
        
    Returns:
        Validation results with missing sections
    """
    # Convert structure to a simple list of section titles for analysis
    section_titles = []
    
    def extract_titles(nodes):
        for node in nodes:
            section_titles.append(node.get('title', ''))
            if 'nodes' in node and node['nodes']:
                extract_titles(node['nodes'])
    
    extract_titles(structure)
    
    prompt = f"""
    You are an expert in SEC regulatory filings. Your task is to validate if all required sections for a 10-K filing are present.
    
    Document sections:
    {json.dumps(section_titles)}
    
    Please analyze the sections and determine:
    1. If all required sections for a 10-K filing are present
    2. Which required sections are missing, if any
    3. Suggestions for misclassified sections (if a required section appears to be present but under a different name)
    
    Reply format:
    {{
        "is_complete": true/false,
        "missing_sections": ["section1", "section2", ...],
        "misclassified_sections": [
            {{
                "standard_name": "standard section name",
                "likely_match": "section title from the document that likely corresponds"
            }}
        ],
        "compliance_score": 0.0-1.0
    }}
    Directly return the final JSON structure. Do not output anything else.
    """
    
    response = await ChatGPT_API_async(model=model, prompt=prompt)
    return extract_json(response)

################### Footnote and Reference Linking #########################################################

async def detect_footnote_references(page_text, model=None):
    """
    Detect footnote references in the text.
    
    Args:
        page_text: Text content of the page
        model: LLM model to use
        
    Returns:
        Detected footnote references
    """
    prompt = f"""
    You are a document analysis expert. Your task is to detect footnote references in the given text.
    
    Page content:
    {page_text}
    
    Please identify all footnote references in the text. These could be:
    1. Superscript numbers or symbols
    2. Numbers or symbols in parentheses
    3. Any other special characters used as reference markers
    
    For each reference, provide:
    1. The reference marker
    2. The position in the text
    3. The context (surrounding text)
    
    Reply format:
    {{
        "footnote_references": [
            {{
                "marker": "reference marker",
                "position": "approximate position in text",
                "context": "surrounding text"
            }}
        ]
    }}
    Directly return the final JSON structure. Do not output anything else.
    """
    
    response = await ChatGPT_API_async(model=model, prompt=prompt)
    return extract_json(response)

async def extract_footnote_content(page_text, model=None):
    """
    Extract footnote content from the text.
    
    Args:
        page_text: Text content of the page
        model: LLM model to use
        
    Returns:
        Extracted footnote content
    """
    prompt = f"""
    You are a document analysis expert. Your task is to extract footnote content from the given text.
    
    Page content:
    {page_text}
    
    Please identify and extract all footnote content. Footnotes are typically found at the bottom of the page or at the end of a section.
    
    For each footnote, provide:
    1. The footnote marker
    2. The complete footnote text
    
    Reply format:
    {{
        "footnotes": [
            {{
                "marker": "footnote marker",
                "content": "footnote text"
            }}
        ]
    }}
    Directly return the final JSON structure. Do not output anything else.
    """
    
    response = await ChatGPT_API_async(model=model, prompt=prompt)
    return extract_json(response)

async def build_reference_graph(footnote_references, footnote_contents, model=None):
    """
    Build a graph connecting footnote references to their content.
    
    Args:
        footnote_references: Detected footnote references
        footnote_contents: Extracted footnote contents
        model: LLM model to use
        
    Returns:
        Reference graph connecting references to content
    """
    # Create a mapping from footnote markers to their content
    marker_to_content = {
        footnote['marker']: footnote['content']
        for footnote in footnote_contents.get('footnotes', [])
    }
    
    # Create the reference graph
    graph = []
    for reference in footnote_references.get('footnote_references', []):
        marker = reference['marker']
        graph.append({
            'source': {
                'marker': marker,
                'context': reference['context'],
                'position': reference['position']
            },
            'target': {
                'content': marker_to_content.get(marker, 'Content not found')
            }
        })
    
    return {
        'reference_graph': graph
    }

################### Enhanced TOC Processing for Financial Documents #########################################################

async def enhance_toc_for_financial_document(toc_content, model=None):
    """
    Enhance the table of contents processing specifically for financial documents.
    
    Args:
        toc_content: Extracted table of contents
        model: LLM model to use
        
    Returns:
        Enhanced TOC with financial document specific improvements
    """
    prompt = f"""
    You are a financial document structure expert. Your task is to enhance the table of contents for a financial document.
    
    Table of contents:
    {toc_content}
    
    Please analyze the TOC and:
    1. Identify standard financial document sections (MD&A, Risk Factors, Financial Statements, etc.)
    2. Standardize section titles to match SEC terminology
    3. Identify any missing key sections that should be present in a financial document
    4. Suggest a hierarchical structure that follows standard financial document organization
    
    Reply format:
    {{
        "enhanced_toc": [
            {{
                "title": "standardized section title",
                "original_title": "original title from document",
                "section_type": "type of financial section",
                "importance": "high/medium/low",
                "expected_content": "brief description of expected content"
            }}
        ],
        "missing_sections": [
            {{
                "title": "missing section title",
                "importance": "high/medium/low",
                "expected_location": "where this section would typically appear"
            }}
        ]
    }}
    Directly return the final JSON structure. Do not output anything else.
    """
    
    response = await ChatGPT_API_async(model=model, prompt=prompt)
    return extract_json(response)

################### Financial Term Detection #########################################################

async def extract_financial_terms(page_text, model=None):
    """
    Extract financial terms and their definitions from the text.
    
    Args:
        page_text: Text content of the page
        model: LLM model to use
        
    Returns:
        Extracted financial terms and definitions
    """
    prompt = f"""
    You are a financial terminology expert. Your task is to extract financial terms and their definitions from the given text.
    
    Page content:
    {page_text}
    
    Please identify all financial terms and their definitions. Focus on:
    1. Explicitly defined terms (e.g., "Term X means...")
    2. Industry-specific financial terminology
    3. Acronyms and abbreviations related to finance
    4. Key financial metrics mentioned
    
    For each term, provide:
    1. The term itself
    2. The definition or explanation provided in the text
    3. The context in which it appears
    
    Reply format:
    {{
        "financial_terms": [
            {{
                "term": "financial term",
                "definition": "definition from text",
                "context": "surrounding context",
                "is_explicit_definition": true/false
            }}
        ]
    }}
    Directly return the final JSON structure. Do not output anything else.
    """
    
    response = await ChatGPT_API_async(model=model, prompt=prompt)
    return extract_json(response)

################### Integration with Page Index #########################################################

async def process_financial_document(page_list, start_index=1, model=None, logger=None):
    """
    Process a financial document with specialized features for SEC 10K documents.
    
    Args:
        page_list: List of pages with their content
        start_index: Starting page index
        model: LLM model to use
        logger: Logger instance
        
    Returns:
        Enhanced document structure with financial-specific features
    """
    if logger:
        logger.info("Processing financial document with specialized features")
    
    # Initialize structure to store results
    financial_structure = []
    
    # Process each page for financial features
    for i, (page_text, _) in enumerate(page_list):
        page_index = i + start_index
        
        # Detect financial tables
        tables = await detect_financial_tables(page_text, model=model)
        
        # Extract footnotes
        footnotes = await extract_footnote_content(page_text, model=model)
        
        # Extract financial terms
        terms = await extract_financial_terms(page_text, model=model)
        
        # Add to structure
        financial_structure.append({
            'page_index': page_index,
            'tables': tables.get('tables', []) if tables.get('tables_detected', False) else [],
            'footnotes': footnotes.get('footnotes', []),
            'financial_terms': terms.get('financial_terms', [])
        })
    
    return financial_structure
