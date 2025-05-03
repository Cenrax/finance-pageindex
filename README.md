# finance-pageindex

A powerful system for processing and analyzing financial documents, with special focus on SEC 10-K filings. This tool extracts structured information from complex financial documents, identifies key sections, recognizes financial tables, extracts footnotes, and provides summaries of document content.

https://github.com/Cenrax/finance-pageindex

## Features

- **Document Structure Extraction**: Automatically detects and extracts the hierarchical structure of financial documents
- **Table of Contents Processing**: Identifies and processes TOC with or without page numbers
- **Financial Table Recognition**: Detects, extracts, and structures financial tables (balance sheets, income statements, etc.)
- **Regulatory Section Identification**: Recognizes standard regulatory sections in SEC filings
- **Footnote Processing**: Extracts footnotes and builds reference graphs connecting them to their mentions
- **Financial Term Extraction**: Identifies and defines financial terminology used in the document
- **Structure Verification**: Validates that extracted sections match the actual document content
- **Content Summarization**: Generates concise summaries of document sections

## Architecture Diagram

```mermaid
flowchart TB
    subgraph Client["Client Interface"]
        UI["User Interface"]
        API["API"]
    end

    subgraph Core["Core Processing"]
        page_index_main["page_index_main()"]
        financial_document_parser["financial_document_parser()"]
        tree_parser["tree_parser()"]
    end
    
    subgraph TOC["Table of Contents Processing"]
        direction TB
        check_toc["check_toc()"]
        toc_detector["toc_detector_single_page()"]
        find_toc_pages["find_toc_pages()"]
        toc_extractor["toc_extractor()"]
        toc_transformer["toc_transformer()"]
        process_toc_with_page_numbers["process_toc_with_page_numbers()"]
        process_toc_no_page_numbers["process_toc_no_page_numbers()"]
        process_no_toc["process_no_toc()"]
        enhance_toc["enhance_toc_for_financial_document()"]
    end
    
    subgraph Verification["Structure Verification"]
        verify_toc["verify_toc()"]
        check_title_appearance["check_title_appearance()"]
        check_title_appearance_in_start["check_title_appearance_in_start()"]
        fix_incorrect_toc["fix_incorrect_toc()"]
    end
    
    subgraph Financial["Financial Feature Extraction"]
        detect_financial_tables["detect_financial_tables()"]
        extract_table_structure["extract_table_structure()"]
        table_to_json["table_to_json()"]
        identify_regulatory_section["identify_regulatory_section()"]
        validate_regulatory_completeness["validate_regulatory_completeness()"]
        extract_financial_terms["extract_financial_terms()"]
        detect_footnote_references["detect_footnote_references()"]
        extract_footnote_content["extract_footnote_content()"]
        build_reference_graph["build_reference_graph()"]
        process_financial_features["process_financial_features_for_structure()"]
    end
    
    subgraph Utils["Utilities"]
        ChatGPT_API["ChatGPT_API()"]
        ChatGPT_API_async["ChatGPT_API_async()"]
        extract_json["extract_json()"]
        get_page_tokens["get_page_tokens()"]
        post_processing["post_processing()"]
        add_node_text["add_node_text()"]
        generate_summaries["generate_summaries_for_structure()"]
    end
    
    subgraph PDF["PDF Processing"]
        PyPDF2["PyPDF2"]
        PyMuPDF["PyMuPDF"]
    end
    
    subgraph LLM["Language Models"]
        GPT4["GPT-4"]
    end
    
    %% Connections
    Client --> Core
    Core --> TOC
    Core --> Financial
    Core --> Verification
    Core --> Utils
    
    TOC --> Verification
    TOC --> Utils
    Financial --> Utils
    Verification --> Utils
    
    Utils --> LLM
    Utils --> PDF
    
    %% Main flow
    page_index_main --> check_if_financial_document
    check_if_financial_document --> financial_document_parser
    financial_document_parser --> tree_parser
    financial_document_parser --> process_financial_features
    tree_parser --> check_toc
    check_toc --> toc_extractor
    toc_extractor --> meta_processor
    
    meta_processor --> process_toc_with_page_numbers
    meta_processor --> process_toc_no_page_numbers
    meta_processor --> process_no_toc
    
    process_toc_with_page_numbers --> verify_toc
    process_toc_no_page_numbers --> verify_toc
    process_no_toc --> verify_toc
    
    verify_toc -- "errors detected" --> fix_incorrect_toc
    
    meta_processor -- "financial document" --> enhance_toc
    
    process_financial_features --> detect_financial_tables
    process_financial_features --> identify_regulatory_section
    process_financial_features --> extract_financial_terms
    process_financial_features --> detect_footnote_references
    
    detect_financial_tables --> extract_table_structure
    extract_table_structure --> table_to_json
    detect_footnote_references --> extract_footnote_content
    extract_footnote_content --> build_reference_graph
    
    classDef core fill:#f9f,stroke:#333,stroke-width:2px;
    classDef financial fill:#bbf,stroke:#333,stroke-width:1px;
    classDef toc fill:#bfb,stroke:#333,stroke-width:1px;
    classDef utils fill:#fbb,stroke:#333,stroke-width:1px;
    
    class Core core;
    class Financial financial;
    class TOC toc;
    class Utils utils;

```

## Installation

```bash
# Clone the repository
git clone https://github.com/Cenrax/finance-pageindex.git
cd finance-pageindex

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your OpenAI API key
```

## Usage

### Basic Usage

```python
from finance_pageindex import page_index

# Process a PDF document
result = page_index("path/to/document.pdf")

# Access the document structure
structure = result['structure']

# Print document sections
for section in structure:
    print(f"Section: {section['title']}")
    if 'summary' in section:
        print(f"Summary: {section['summary']}")
    print()
```

### Advanced Configuration

You can configure the processing with additional parameters:

```python
result = page_index(
    "path/to/document.pdf",
    model="gpt-4o-",
    toc_check_page_num=20,
    max_page_num_each_node=50,
    max_token_num_each_node=100000,
    if_add_node_id="yes",
    if_add_node_summary="yes",
    if_add_doc_description="yes",
    if_add_node_text="yes",
    process_financial_features="yes",
    extract_tables="yes",
    extract_footnotes="yes"
)
```

## Architecture

The system is designed with a modular architecture:

1. **Core Processing**
   - Document ingestion and page tokenization
   - Structure extraction and verification
   - Financial feature detection

2. **Table of Contents (TOC) Processing**
   - TOC detection and extraction
   - Structure transformation with page mapping

3. **Financial Features**
   - Table detection and extraction
   - Regulatory section identification
   - Financial term extraction
   - Footnote processing

4. **Utilities**
   - PDF processing with PyPDF2/PyMuPDF
   - LLM integration with OpenAI's GPT models
   - Structure manipulation and validation

## Dependencies

- Python 3.8+
- OpenAI API access
- PyPDF2/PyMuPDF for PDF processing
- tiktoken for token counting
- asyncio for concurrent processing

## Contributing

Contributions are welcome! Please check the development checklist for areas that need improvement.

## License

[MIT License](LICENSE)

## Development Status Checklist

- [x] Core document structure extraction
- [x] Table of Contents processing
- [x] Financial table detection and extraction
- [x] Regulatory section identification
- [x] Footnote processing
- [x] Financial term extraction
- [x] Structure verification and correction
- [x] Section summarization
- [ ] User-friendly command line interface
- [ ] Web API for remote document processing
- [ ] Performance optimization for large documents
- [ ] Support for non-PDF document formats
- [ ] Browser-based visualization of extracted structure
- [ ] Export to multiple formats (JSON, CSV, Excel)
- [ ] Batch processing capability
- [ ] Improved error handling and reporting
- [ ] Integration with financial analysis tools
- [ ] Support for additional financial document types
- [ ] Multilingual document support
- [ ] Comprehensive test suite
