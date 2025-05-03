#!/usr/bin/env python3

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from pageindex.page_index import page_index
from io import BytesIO
from PyPDF2 import PdfReader, PdfWriter

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process financial SEC 10K documents with enhanced page indexing')
    parser.add_argument('input_file', type=str, help='Path to the input PDF file')
    parser.add_argument('--output', type=str, help='Path to save the output JSON file (default: auto-generated)')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='OpenAI model to use')
    parser.add_argument('--no-tables', action='store_true', help='Disable financial table extraction')
    parser.add_argument('--no-footnotes', action='store_true', help='Disable footnote extraction')
    parser.add_argument('--no-summaries', action='store_true', help='Disable node summaries')
    parser.add_argument('--safe-mode', action='store_true', help='Enable safe mode for more robust processing')
    parser.add_argument('--first-pages', type=int, help='Process only the first N pages of the PDF')
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists() or not input_path.is_file() or input_path.suffix.lower() != '.pdf':
        print(f"Error: '{args.input_file}' is not a valid PDF file.")
        sys.exit(1)
    
    # Determine output file path
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = input_path.with_name(f"{input_path.stem}_financial_index_{timestamp}.json")
    
    # Prepare document (slice first N pages if requested)
    if args.first_pages:
        reader = PdfReader(str(input_path))
        writer = PdfWriter()
        total_pages = len(reader.pages)
        max_pages = min(args.first_pages, total_pages)
        for i in range(max_pages):
            writer.add_page(reader.pages[i])
        buffer = BytesIO()
        writer.write(buffer)
        buffer.seek(0)
        doc_to_process = buffer
    else:
        doc_to_process = str(input_path)

    print(f"Processing financial document: {input_path} (first {args.first_pages if args.first_pages else 'all'} pages)")
    print(f"Using model: {args.model}")
    
    try:
        # Process the document with financial features
        result = page_index(
            doc=doc_to_process,
            model=args.model,
            process_financial_features='yes',
            extract_tables='no' if args.no_tables else 'yes',
            extract_footnotes='no' if args.no_footnotes else 'yes',
            if_add_node_summary='no' if args.no_summaries else 'yes',
            if_add_node_id='yes',
            if_add_doc_description='yes',
            # Use a smaller TOC check page number in safe mode
            toc_check_page_num=10 if args.safe_mode else None,
            # Use smaller max page and token numbers in safe mode
            max_page_num_each_node=5 if args.safe_mode else None,
            max_token_num_each_node=10000 if args.safe_mode else None
        )
    except Exception as e:
        print(f"\nError processing document: {str(e)}")
        print("\nTry running with --safe-mode for more robust processing.")
        sys.exit(1)
    
    # Save the result to the output file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\nProcessing complete!")
    print(f"Document name: {result.get('doc_name', 'Unknown')}")
    print(f"Is financial document: {result.get('is_financial_document', False)}")
    if 'doc_description' in result:
        print(f"Document description: {result['doc_description']}")
    print(f"Output saved to: {output_path}")
    
    # Print structure statistics
    structure = result.get('structure', [])
    total_sections = count_sections(structure)
    print(f"Total sections: {total_sections}")
    
    # Print financial features summary if it's a financial document
    if result.get('is_financial_document', False):
        print("\nFinancial Features Summary:")
        financial_stats = get_financial_stats(structure)
        print(f"  - Tables detected: {financial_stats['tables']}")
        print(f"  - Footnotes detected: {financial_stats['footnotes']}")
        print(f"  - Financial terms extracted: {financial_stats['terms']}")
        print(f"  - Regulatory sections identified: {financial_stats['regulatory']}")

def count_sections(structure):
    """Count the total number of sections in the structure."""
    if not structure:
        return 0
    
    count = len(structure)
    for node in structure:
        if isinstance(node, dict) and 'nodes' in node and node['nodes']:
            count += count_sections(node['nodes'])
    
    return count

def get_financial_stats(structure):
    """Get statistics about financial features in the document."""
    stats = {
        'tables': 0,
        'footnotes': 0,
        'terms': 0,
        'regulatory': 0
    }
    
    def process_node(node):
        if isinstance(node, dict):
            # Check for metadata
            if 'metadata' in node:
                # Count tables
                if 'tables' in node['metadata']:
                    stats['tables'] += len(node['metadata']['tables'])
                
                # Count footnotes
                if 'footnotes' in node['metadata'] and 'contents' in node['metadata']['footnotes']:
                    stats['footnotes'] += len(node['metadata']['footnotes']['contents'])
                
                # Count financial terms
                if 'financial_terms' in node['metadata']:
                    stats['terms'] += len(node['metadata']['financial_terms'])
                
                # Count regulatory sections
                if 'regulatory_info' in node['metadata'] and node['metadata']['regulatory_info'].get('is_regulatory', False):
                    stats['regulatory'] += 1
            
            # Process child nodes
            if 'nodes' in node and node['nodes']:
                for child in node['nodes']:
                    process_node(child)
    
    # Process all top-level nodes
    for node in structure:
        process_node(node)
    
    return stats

if __name__ == "__main__":
    main()
