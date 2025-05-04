import os
import json
import copy
import math
import random
import re
from .utils import *
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from .features import (
    detect_financial_tables, extract_table_structure, table_to_json,
    identify_regulatory_section, validate_regulatory_completeness,
    detect_footnote_references, extract_footnote_content, build_reference_graph,
    enhance_toc_for_financial_document, extract_financial_terms
)


################### check title in page #########################################################
async def check_title_appearance(item, page_list, start_index=1, model=None):    
    title=item['title']
    if 'physical_index' not in item or item['physical_index'] is None:
        return {'list_index': item.get('list_index'), 'answer': 'no', 'title':title, 'page_number': None}
    
    
    page_number = item['physical_index']
    
    # Add bounds checking to prevent index errors
    if page_number < start_index or (page_number - start_index) >= len(page_list):
        return {'list_index': item.get('list_index'), 'answer': 'no', 'title':title, 'page_number': page_number, 'error': 'page_out_of_range'}
    
    page_text = page_list[page_number-start_index][0]

    
    prompt = f"""
    Your job is to check if the given section appears or starts in the given page_text.

    Note: do fuzzy matching, ignore any space inconsistency in the page_text.

    The given section title is {title}.
    The given page_text is {page_text}.
    
    Reply format:
    {{
        
        "thinking": <why do you think the section appears or starts in the page_text>
        "answer": "yes or no" (yes if the section appears or starts in the page_text, no otherwise)
    }}
    Directly return the final JSON structure. Do not output anything else."""

    response = await ChatGPT_API_async(model=model, prompt=prompt)
    response = extract_json(response)
    if 'answer' in response:
        answer = response['answer']
    else:
        answer = 'no'
    return {'list_index': item['list_index'], 'answer': answer, 'title': title, 'page_number': page_number}


async def check_title_appearance_in_start(title, page_text, model=None, logger=None):    
    prompt = f"""
    You will be given the current section title and the current page_text.
    Your job is to check if the current section starts in the beginning of the given page_text.
    If there are other contents before the current section title, then the current section does not start in the beginning of the given page_text.
    If the current section title is the first content in the given page_text, then the current section starts in the beginning of the given page_text.

    Note: do fuzzy matching, ignore any space inconsistency in the page_text.

    The given section title is {title}.
    The given page_text is {page_text}.
    
    reply format:
    {{
        "thinking": <why do you think the section appears or starts in the page_text>
        "start_begin": "yes or no" (yes if the section starts in the beginning of the page_text, no otherwise)
    }}
    Directly return the final JSON structure. Do not output anything else."""

    response = await ChatGPT_API_async(model=model, prompt=prompt)
    response = extract_json(response)
    if logger:
        logger.info(f"Response: {response}")
    return response.get("start_begin", "no")


async def check_title_appearance_in_start_concurrent(structure, page_list, model=None, logger=None):
    if logger:
        logger.info("Checking title appearance in start concurrently")
    
    # skip items without physical_index
    for item in structure:
        if item.get('physical_index') is None:
            item['appear_start'] = 'no'

    # only for items with valid physical_index
    tasks = []
    valid_items = []
    for item in structure:
        if item.get('physical_index') is not None:
            # Add bounds checking to prevent index errors
            physical_index = item['physical_index']
            if physical_index < 1 or (physical_index - 1) >= len(page_list):
                item['appear_start'] = 'no'
                if logger:
                    logger.warning(f"Page index out of range for title: {item['title']}, index: {physical_index}")
                continue
                
            page_text = page_list[physical_index - 1][0]
            tasks.append(check_title_appearance_in_start(item['title'], page_text, model=model, logger=logger))
            valid_items.append(item)

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for item, result in zip(valid_items, results):
        if isinstance(result, Exception):
            if logger:
                logger.error(f"Error checking start for {item['title']}: {result}")
            item['appear_start'] = 'no'
        else:
            item['appear_start'] = result

    return structure


def toc_detector_single_page(content, model=None):
    prompt = f"""
    Your job is to detect if there is a table of content provided in the given text.

    Given text: {content}

    return the following JSON format:
    {{
        "thinking": <why do you think there is a table of content in the given text>
        "toc_detected": "<yes or no>",
    }}

    Directly return the final JSON structure. Do not output anything else.
    Please note: abstract,summary, notation list, figure list, table list, etc. are not table of contents."""

    response = ChatGPT_API(model=model, prompt=prompt)
    # print('response', response)
    json_content = extract_json(response)    
    return json_content['toc_detected']


def check_if_toc_extraction_is_complete(content, toc, model=None):
    prompt = f"""
    You are given a partial document  and a  table of contents.
    Your job is to check if the  table of contents is complete, which it contains all the main sections in the partial document.

    Reply format:
    {{
        "thinking": <why do you think the table of contents is complete or not>
        "completed": "yes" or "no"
    }}
    Directly return the final JSON structure. Do not output anything else."""

    prompt = prompt + '\n Document:\n' + content + '\n Table of contents:\n' + toc
    response = ChatGPT_API(model=model, prompt=prompt)
    json_content = extract_json(response)
    return json_content['completed']


def check_if_toc_transformation_is_complete(content, toc, model=None):
    prompt = f"""
    You are given a raw table of contents and a  table of contents.
    Your job is to check if the  table of contents is complete.

    Reply format:
    {{
        "thinking": <why do you think the cleaned table of contents is complete or not>
        "completed": "yes" or "no"
    }}
    Directly return the final JSON structure. Do not output anything else."""

    prompt = prompt + '\n Raw Table of contents:\n' + content + '\n Cleaned Table of contents:\n' + toc
    response = ChatGPT_API(model=model, prompt=prompt)
    json_content = extract_json(response)
    return json_content['completed']

def extract_toc_content(content, model=None):
    prompt = f"""
    Your job is to extract the full table of contents from the given text, replace ... with :

    Given text: {content}

    Directly return the full table of contents content. Do not output anything else."""

    response, finish_reason = ChatGPT_API_with_finish_reason(model=model, prompt=prompt)
    
    if_complete = check_if_toc_transformation_is_complete(content, response, model)
    if if_complete == "yes" and finish_reason == "finished":
        return response
    
    chat_history = [
        {"role": "user", "content": prompt}, 
        {"role": "assistant", "content": response},    
    ]
    prompt = f"""please continue the generation of table of contents , directly output the remaining part of the structure"""
    new_response, finish_reason = ChatGPT_API_with_finish_reason(model=model, prompt=prompt, chat_history=chat_history)
    response = response + new_response
    if_complete = check_if_toc_transformation_is_complete(content, response, model)
    
    while not (if_complete == "yes" and finish_reason == "finished"):
        chat_history = [
            {"role": "user", "content": prompt}, 
            {"role": "assistant", "content": response},    
        ]
        prompt = f"""please continue the generation of table of contents , directly output the remaining part of the structure"""
        new_response, finish_reason = ChatGPT_API_with_finish_reason(model=model, prompt=prompt, chat_history=chat_history)
        response = response + new_response
        if_complete = check_if_toc_transformation_is_complete(content, response, model)
        
        # Optional: Add a maximum retry limit to prevent infinite loops
        if len(chat_history) > 5:  # Arbitrary limit of 10 attempts
            raise Exception('Failed to complete table of contents after maximum retries')
    
    return response

def detect_page_index(toc_content, model=None):
    print('start detect_page_index')
    prompt = f"""
    You will be given a table of contents.

    Your job is to detect if there are page numbers/indices given within the table of contents.

    Given text: {toc_content}

    Reply format:
    {{
        "thinking": <why do you think there are page numbers/indices given within the table of contents>
        "page_index_given_in_toc": "<yes or no>"
    }}
    Directly return the final JSON structure. Do not output anything else."""

    response = ChatGPT_API(model=model, prompt=prompt)
    json_content = extract_json(response)
    return json_content['page_index_given_in_toc']

def toc_extractor(page_list, toc_page_list, model):
    def transform_dots_to_colon(text):
        text = re.sub(r'\.{5,}', ': ', text)
        # Handle dots separated by spaces
        text = re.sub(r'(?:\. ){5,}\.?', ': ', text)
        return text
    
    toc_content = ""
    for page_index in toc_page_list:
        toc_content += page_list[page_index][0]
    toc_content = transform_dots_to_colon(toc_content)
    has_page_index = detect_page_index(toc_content, model=model)
    
    return {
        "toc_content": toc_content,
        "page_index_given_in_toc": has_page_index
    }




def toc_index_extractor(toc, content, model=None):
    print('start toc_index_extractor')
    tob_extractor_prompt = """
    You are given a table of contents in a json format and several pages of a document, your job is to add the physical_index to the table of contents in the json format.

    The provided pages contains tags like <physical_index_X> and <physical_index_X> to indicate the physical location of the page X.

    The structure variable is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

    The response should be in the following JSON format: 
    [
        {
            "structure": <structure index, "x.x.x" or None> (string),
            "title": <title of the section>,
            "physical_index": "<physical_index_X>" (keep the format)
        },
        ...
    ]

    Only add the physical_index to the sections that are in the provided pages.
    If the section is not in the provided pages, do not add the physical_index to it.
    Directly return the final JSON structure. Do not output anything else."""

    prompt = tob_extractor_prompt + '\nTable of contents:\n' + str(toc) + '\nDocument pages:\n' + content
    response = ChatGPT_API(model=model, prompt=prompt)
    json_content = extract_json(response)    
    return json_content



def toc_transformer(toc_content, model=None):
    print('start toc_transformer')
    init_prompt = """
    You are given a table of contents, You job is to transform the whole table of content into a JSON format included table_of_contents.

    structure is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

    The response should be in the following JSON format: 
    {
    table_of_contents: [
        {
            "structure": <structure index, "x.x.x" or None> (string),
            "title": <title of the section>,
            "page": <page number or None>,
        },
        ...
        ],
    }
    You should transform the full table of contents in one go.
    Directly return the final JSON structure, do not output anything else. """

    prompt = init_prompt + '\n Given table of contents\n:' + toc_content
    last_complete, finish_reason = ChatGPT_API_with_finish_reason(model=model, prompt=prompt)
    if_complete = check_if_toc_transformation_is_complete(toc_content, last_complete, model)
    if if_complete == "yes" and finish_reason == "finished":
        last_complete = extract_json(last_complete)
        cleaned_response=convert_page_to_int(last_complete['table_of_contents'])
        return cleaned_response
    
    last_complete = get_json_content(last_complete)
    while not (if_complete == "yes" and finish_reason == "finished"):
        position = last_complete.rfind('}')
        if position != -1:
            last_complete = last_complete[:position+2]
        prompt = f"""
        Your task is to continue the table of contents json structure, directly output the remaining part of the json structure.
        The response should be in the following JSON format: 

        The raw table of contents json structure is:
        {toc_content}

        The incomplete transformed table of contents json structure is:
        {last_complete}

        Please continue the json structure, directly output the remaining part of the json structure."""

        new_complete, finish_reason = ChatGPT_API_with_finish_reason(model=model, prompt=prompt)

        if new_complete.startswith('```json'):
            new_complete =  get_json_content(new_complete)
            last_complete = last_complete+new_complete

        if_complete = check_if_toc_transformation_is_complete(toc_content, last_complete, model)
        

    last_complete = json.loads(last_complete)

    cleaned_response=convert_page_to_int(last_complete['table_of_contents'])
    return cleaned_response
    



def find_toc_pages(start_page_index, page_list, opt, logger=None):
    print('start find_toc_pages')
    last_page_is_yes = False
    toc_page_list = []
    i = start_page_index
    
    while i < len(page_list):
        # Only check beyond max_pages if we're still finding TOC pages
        if i >= opt.toc_check_page_num and not last_page_is_yes:
            break
        detected_result = toc_detector_single_page(page_list[i][0],model=opt.model)
        if detected_result == 'yes':
            if logger:
                logger.info(f'Page {i} has toc')
            toc_page_list.append(i)
            last_page_is_yes = True
        elif detected_result == 'no' and last_page_is_yes:
            if logger:
                logger.info(f'Found the last page with toc: {i-1}')
            break
        i += 1
    
    if not toc_page_list and logger:
        logger.info('No toc found')
        
    return toc_page_list

def remove_page_number(data):
    if isinstance(data, dict):
        data.pop('page_number', None)  
        for key in list(data.keys()):
            if 'nodes' in key:
                remove_page_number(data[key])
    elif isinstance(data, list):
        for item in data:
            remove_page_number(item)
    return data

def extract_matching_page_pairs(toc_page, toc_physical_index, start_page_index):
    pairs = []
    for phy_item in toc_physical_index:
        title = phy_item.get('title')
        phys = phy_item.get('physical_index')
        # Only proceed with numeric physical_index
        try:
            phys_int = int(phys)
        except (ValueError, TypeError):
            continue
        if phys_int < start_page_index:
            continue
        for page_item in toc_page:
            if title == page_item.get('title'):
                pairs.append({
                    'title': title,
                    'page': page_item.get('page'),
                    'physical_index': phys_int
                })
    return pairs


def calculate_page_offset(pairs):
    differences = []
    for pair in pairs:
        try:
            physical_index = pair['physical_index']
            page_number = pair['page']
            difference = physical_index - page_number
            differences.append(difference)
        except (KeyError, TypeError):
            continue
    
    if not differences:
        return None
    
    difference_counts = {}
    for diff in differences:
        difference_counts[diff] = difference_counts.get(diff, 0) + 1
    
    most_common = max(difference_counts.items(), key=lambda x: x[1])[0]
    
    return most_common

def add_page_offset_to_toc_json(data, offset):
    for i in range(len(data)):
        if data[i].get('page') is not None and isinstance(data[i]['page'], int):
            data[i]['physical_index'] = data[i]['page'] + offset
            del data[i]['page']
    
    return data



def page_list_to_group_text(page_contents, token_lengths, max_tokens=20000, overlap_page=1):    
    num_tokens = sum(token_lengths)
    
    if num_tokens <= max_tokens:
        # merge all pages into one text
        page_text = "".join(page_contents)
        return [page_text]
    
    subsets = []
    current_subset = []
    current_token_count = 0

    expected_parts_num = math.ceil(num_tokens / max_tokens)
    average_tokens_per_part = math.ceil(((num_tokens / expected_parts_num) + max_tokens) / 2)
    
    for i, (page_content, page_tokens) in enumerate(zip(page_contents, token_lengths)):
        if current_token_count + page_tokens > average_tokens_per_part:

            subsets.append(''.join(current_subset))
            # Start new subset from overlap if specified
            overlap_start = max(i - overlap_page, 0)
            current_subset = page_contents[overlap_start:i]
            current_token_count = sum(token_lengths[overlap_start:i])
        
        # Add current page to the subset
        current_subset.append(page_content)
        current_token_count += page_tokens

    # Add the last subset if it contains any pages
    if current_subset:
        subsets.append(''.join(current_subset))
    
    print('divide page_list to groups', len(subsets))
    return subsets

def add_page_number_to_toc(part, structure, model=None):
    fill_prompt_seq = """
    You are given an JSON structure of a document and a partial part of the document. Your task is to check if the title that is described in the structure is started in the partial given document.

    The provided text contains tags like <physical_index_X> and <physical_index_X> to indicate the physical location of the page X. 

    If the full target section starts in the partial given document, insert the given JSON structure with the "start": "yes", and "start_index": "<physical_index_X>".

    If the full target section does not start in the partial given document, insert "start": "no",  "start_index": None.

    The response should be in the following format. 
        [
            {
                "structure": <structure index, "x.x.x" or None> (string),
                "title": <title of the section>,
                "start": "<yes or no>",
                "physical_index": "<physical_index_X> (keep the format)" or None
            },
            ...
        ]    
    The given structure contains the result of the previous part, you need to fill the result of the current part, do not change the previous result.
    Directly return the final JSON structure. Do not output anything else."""

    prompt = fill_prompt_seq + f"\n\nCurrent Partial Document:\n{part}\n\nGiven Structure\n{json.dumps(structure, indent=2)}\n"
    current_json_raw = ChatGPT_API(model=model, prompt=prompt)
    json_result = extract_json(current_json_raw)
    
    for item in json_result:
        if 'start' in item:
            del item['start']
    return json_result


def remove_first_physical_index_section(text):
    """
    Removes the first section between <physical_index_X> and <physical_index_X> tags,
    and returns the remaining text.
    """
    pattern = r'<physical_index_\d+>.*?<physical_index_\d+>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        # Remove the first matched section
        return text.replace(match.group(0), '', 1)
    return text

### add verify completeness
def generate_toc_continue(toc_content, part, model="gpt-4o-2024-11-20"):
    print('start generate_toc_continue')
    prompt = """
    You are an expert in extracting hierarchical tree structure.
    You are given a tree structure of the previous part and the text of the current part.
    Your task is to continue the tree structure from the previous part to include the current part.

    The structure variable is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

    For the title, you need to extract the original title from the text, only fix the space inconsistency.

    The provided text contains tags like <physical_index_X> and <physical_index_X> to indicate the start and end of page X. \
    
    For the physical_index, you need to extract the physical index of the start of the section from the text. Keep the <physical_index_X> format.

    The response should be in the following format. 
        [
            {
                "structure": <structure index, "x.x.x"> (string),
                "title": <title of the section, keep the original title>,
                "physical_index": "<physical_index_X> (keep the format)"
            },
            ...
        ]    

    Directly return the additional part of the final JSON structure. Do not output anything else."""

    prompt = prompt + '\nGiven text\n:' + part + '\nPrevious tree structure\n:' + json.dumps(toc_content, indent=2)
    response, finish_reason = ChatGPT_API_with_finish_reason(model=model, prompt=prompt)
    if finish_reason == 'finished':
        return extract_json(response)
    else:
        raise Exception(f'finish reason: {finish_reason}')
    
### add verify completeness
def generate_toc_init(part, model=None):
    print('start generate_toc_init')
    prompt = """
    You are an expert in extracting hierarchical tree structure, your task is to generate the tree structure of the document.

    The structure variable is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

    For the title, you need to extract the original title from the text, only fix the space inconsistency.

    The provided text contains tags like <physical_index_X> and <physical_index_X> to indicate the start and end of page X. 

    For the physical_index, you need to extract the physical index of the start of the section from the text. Keep the <physical_index_X> format.

    The response should be in the following format. 
        [
            {{
                "structure": <structure index, "x.x.x"> (string),
                "title": <title of the section, keep the original title>,
                "physical_index": "<physical_index_X> (keep the format)"
            }},
            
        ],


    Directly return the final JSON structure. Do not output anything else."""

    prompt = prompt + '\nGiven text\n:' + part
    response, finish_reason = ChatGPT_API_with_finish_reason(model=model, prompt=prompt)

    if finish_reason == 'finished':
         return extract_json(response)
    else:
        raise Exception(f'finish reason: {finish_reason}')

def process_no_toc(page_list, start_index=1, model=None, logger=None):
    page_contents=[]
    token_lengths=[]
    for page_index in range(start_index, start_index+len(page_list)):
        page_text = f"<physical_index_{page_index}>\n{page_list[page_index-start_index][0]}\n<physical_index_{page_index}>\n\n"
        page_contents.append(page_text)
        token_lengths.append(count_tokens(page_text, model))
    group_texts = page_list_to_group_text(page_contents, token_lengths)
    logger.info(f'len(group_texts): {len(group_texts)}')

    toc_with_page_number= generate_toc_init(group_texts[0], model)
    for group_text in group_texts[1:]:
        toc_with_page_number_additional = generate_toc_continue(toc_with_page_number, group_text, model)    
        toc_with_page_number.extend(toc_with_page_number_additional)
    logger.info(f'generate_toc: {toc_with_page_number}')

    toc_with_page_number = convert_physical_index_to_int(toc_with_page_number)
    logger.info(f'convert_physical_index_to_int: {toc_with_page_number}')

    return toc_with_page_number

def process_toc_no_page_numbers(toc_content, toc_page_list, page_list,  start_index=1, model=None, logger=None):
    page_contents=[]
    token_lengths=[]
    toc_content = toc_transformer(toc_content, model)
    logger.info(f'toc_transformer: {toc_content}')
    for page_index in range(start_index, start_index+len(page_list)):
        page_text = f"<physical_index_{page_index}>\n{page_list[page_index-start_index][0]}\n<physical_index_{page_index}>\n\n"
        page_contents.append(page_text)
        token_lengths.append(count_tokens(page_text, model))
    
    group_texts = page_list_to_group_text(page_contents, token_lengths)
    logger.info(f'len(group_texts): {len(group_texts)}')

    toc_with_page_number=copy.deepcopy(toc_content)
    for group_text in group_texts:
        toc_with_page_number = add_page_number_to_toc(group_text, toc_with_page_number, model)
    logger.info(f'add_page_number_to_toc: {toc_with_page_number}')

    toc_with_page_number = convert_physical_index_to_int(toc_with_page_number)
    logger.info(f'convert_physical_index_to_int: {toc_with_page_number}')

    return toc_with_page_number



def process_toc_with_page_numbers(toc_content, toc_page_list, page_list, toc_check_page_num=None, model=None, logger=None):
    toc_with_page_number = toc_transformer(toc_content, model)
    logger.info(f'toc_with_page_number: {toc_with_page_number}')

    toc_no_page_number = remove_page_number(copy.deepcopy(toc_with_page_number))
    
    start_page_index = toc_page_list[-1] + 1
    main_content = ""
    for page_index in range(start_page_index, min(start_page_index + toc_check_page_num, len(page_list))):
        main_content += f"<physical_index_{page_index+1}>\n{page_list[page_index][0]}\n<physical_index_{page_index+1}>\n\n"

    toc_with_physical_index = toc_index_extractor(toc_no_page_number, main_content, model)
    logger.info(f'toc_with_physical_index: {toc_with_physical_index}')

    toc_with_physical_index = convert_physical_index_to_int(toc_with_physical_index)
    logger.info(f'toc_with_physical_index: {toc_with_physical_index}')

    matching_pairs = extract_matching_page_pairs(toc_with_page_number, toc_with_physical_index, start_page_index)
    logger.info(f'matching_pairs: {matching_pairs}')

    offset = calculate_page_offset(matching_pairs)
    logger.info(f'offset: {offset}')

    toc_with_page_number = add_page_offset_to_toc_json(toc_with_page_number, offset)
    logger.info(f'toc_with_page_number: {toc_with_page_number}')

    toc_with_page_number = process_none_page_numbers(toc_with_page_number, page_list, model=model)
    logger.info(f'toc_with_page_number: {toc_with_page_number}')

    return toc_with_page_number



##check if needed to process none page numbers
def process_none_page_numbers(toc_items, page_list, start_index=1, model=None):
    for i, item in enumerate(toc_items):
        if "physical_index" not in item:
            # logger.info(f"fix item: {item}")
            # Find previous physical_index
            prev_physical_index = 0  # Default if no previous item exists
            for j in range(i - 1, -1, -1):
                if toc_items[j].get('physical_index') is not None:
                    prev_physical_index = toc_items[j]['physical_index']
                    break
            
            # Find next physical_index
            next_physical_index = -1  # Default if no next item exists
            for j in range(i + 1, len(toc_items)):
                if toc_items[j].get('physical_index') is not None:
                    next_physical_index = toc_items[j]['physical_index']
                    break

            page_contents = []
            for page_index in range(prev_physical_index, next_physical_index+1):
                page_text = f"<physical_index_{page_index}>\n{page_list[page_index-start_index][0]}\n<physical_index_{page_index}>\n\n"
                page_contents.append(page_text)

            item_copy = copy.deepcopy(item)
            del item_copy['page']
            result = add_page_number_to_toc(page_contents, item_copy, model)
            if isinstance(result[0]['physical_index'], str) and result[0]['physical_index'].startswith('<physical_index'):
                item['physical_index'] = int(result[0]['physical_index'].split('_')[-1].rstrip('>').strip())
                del item['page']
    
    return toc_items




def check_toc(page_list, opt=None):
    toc_page_list = find_toc_pages(start_page_index=0, page_list=page_list, opt=opt)
    if len(toc_page_list) == 0:
        print('no toc found')
        return {'toc_content': None, 'toc_page_list': [], 'page_index_given_in_toc': 'no'}
    else:
        print('toc found')
        toc_json = toc_extractor(page_list, toc_page_list, opt.model)

        if toc_json['page_index_given_in_toc'] == 'yes':
            print('index found')
            return {'toc_content': toc_json['toc_content'], 'toc_page_list': toc_page_list, 'page_index_given_in_toc': 'yes'}
        else:
            current_start_index = toc_page_list[-1] + 1
            
            while (toc_json['page_index_given_in_toc'] == 'no' and 
                   current_start_index < len(page_list) and 
                   current_start_index < opt.toc_check_page_num):
                
                additional_toc_pages = find_toc_pages(
                    start_page_index=current_start_index,
                    page_list=page_list,
                    opt=opt
                )
                
                if len(additional_toc_pages) == 0:
                    break

                additional_toc_json = toc_extractor(page_list, additional_toc_pages, opt.model)
                if additional_toc_json['page_index_given_in_toc'] == 'yes':
                    print('index found')
                    return {'toc_content': additional_toc_json['toc_content'], 'toc_page_list': additional_toc_pages, 'page_index_given_in_toc': 'yes'}

                else:
                    current_start_index = additional_toc_pages[-1] + 1
            print('index not found')
            return {'toc_content': toc_json['toc_content'], 'toc_page_list': toc_page_list, 'page_index_given_in_toc': 'no'}






################### fix incorrect toc #########################################################
def single_toc_item_index_fixer(section_title, content, model="gpt-4o-2024-11-20"):
    tob_extractor_prompt = """
    You are given a section title and several pages of a document, your job is to find the physical index of the start page of the section in the partial document.

    The provided pages contains tags like <physical_index_X> and <physical_index_X> to indicate the physical location of the page X.

    Reply in a JSON format:
    {
        "thinking": <explain which page, started and closed by <physical_index_X>, contains the start of this section>,
        "physical_index": "<physical_index_X>" (keep the format)
    }
    Directly return the final JSON structure. Do not output anything else."""

    prompt = tob_extractor_prompt + '\nSection Title:\n' + str(section_title) + '\nDocument pages:\n' + content
    response = ChatGPT_API(model=model, prompt=prompt)
    json_content = extract_json(response)
    
    # Handle case where physical_index might not be in expected format
    try:
        physical_index = json_content.get('physical_index', '')
        if not physical_index or not isinstance(physical_index, str):
            return None
            
        # Only proceed if it's in the expected format
        if physical_index.startswith('<physical_index_'):
            return convert_physical_index_to_int(physical_index)
        else:
            return None
    except Exception as e:
        print(f"Error processing physical index: {e}")
        return None



async def fix_incorrect_toc(toc_with_page_number, page_list, incorrect_results, start_index=1, model=None, logger=None):
    print(f'start fix_incorrect_toc with {len(incorrect_results)} incorrect results')
    incorrect_indices = {result['list_index'] for result in incorrect_results}
    
    end_index = len(page_list) + start_index - 1
    
    incorrect_results_and_range_logs = []
    # Helper function to process and check a single incorrect item
    async def process_and_check_item(incorrect_item):
        list_index = incorrect_item['list_index']
        # Find the previous correct item
        prev_correct = None
        for i in range(list_index-1, -1, -1):
            if i not in incorrect_indices:
                prev_correct = toc_with_page_number[i]['physical_index']
                break
        # If no previous correct item found, use start_index
        if prev_correct is None:
            prev_correct = start_index - 1
        
        # Find the next correct item
        next_correct = None
        for i in range(list_index+1, len(toc_with_page_number)):
            if i not in incorrect_indices:
                next_correct = toc_with_page_number[i]['physical_index']
                break
        # If no next correct item found, use end_index
        if next_correct is None:
            next_correct = end_index
        
        incorrect_results_and_range_logs.append({
            'list_index': list_index,
            'title': incorrect_item['title'],
            'prev_correct': prev_correct,
            'next_correct': next_correct
        })

        page_contents=[]
        for page_index in range(prev_correct, next_correct+1):
            page_text = f"<physical_index_{page_index}>\n{page_list[page_index-start_index][0]}\n<physical_index_{page_index}>\n\n"
            page_contents.append(page_text)
        content_range = ''.join(page_contents)
        
        physical_index_int = single_toc_item_index_fixer(incorrect_item['title'], content_range, model)
        
        # Check if the result is correct
        check_item = incorrect_item.copy()
        check_item['physical_index'] = physical_index_int
        check_result = await check_title_appearance(check_item, page_list, start_index, model)

        return {
            'list_index': list_index,
            'title': incorrect_item['title'],
            'physical_index': physical_index_int,
            'is_valid': check_result['answer'] == 'yes'
        }

    # Process incorrect items concurrently
    tasks = [
        process_and_check_item(item)
        for item in incorrect_results
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for item, result in zip(incorrect_results, results):
        if isinstance(result, Exception):
            print(f"Processing item {item} generated an exception: {result}")
            continue
    results = [result for result in results if not isinstance(result, Exception)]

    # Update the toc_with_page_number with the fixed indices and check for any invalid results
    invalid_results = []
    for result in results:
        if result['is_valid']:
            toc_with_page_number[result['list_index']]['physical_index'] = result['physical_index']
        else:
            invalid_results.append({
                'list_index': result['list_index'],
                'title': result['title'],
                'physical_index': result['physical_index'],
            })

    logger.info(f'incorrect_results_and_range_logs: {incorrect_results_and_range_logs}')
    logger.info(f'invalid_results: {invalid_results}')

    return toc_with_page_number, invalid_results



async def fix_incorrect_toc_with_retries(toc_with_page_number, page_list, incorrect_results, start_index=1, max_attempts=3, model=None, logger=None):
    print('start fix_incorrect_toc')
    fix_attempt = 0
    current_toc = toc_with_page_number
    current_incorrect = incorrect_results

    while current_incorrect:
        print(f"Fixing {len(current_incorrect)} incorrect results")
        
        current_toc, current_incorrect = await fix_incorrect_toc(current_toc, page_list, current_incorrect, start_index, model, logger)
                
        fix_attempt += 1
        if fix_attempt >= max_attempts:
            logger.info("Maximum fix attempts reached")
            break
    
    return current_toc, current_incorrect




################### verify toc #########################################################
async def verify_toc(page_list, list_result, start_index=1, N=None, model=None):
    print('start verify_toc')
    # Find the last non-None physical_index
    last_physical_index = None
    for item in reversed(list_result):
        if item.get('physical_index') is not None:
            last_physical_index = item['physical_index']
            break
    
    # Early return if we don't have valid physical indices
    if last_physical_index is None or last_physical_index < len(page_list)/2:
        return 0, []
    
    # Determine which items to check
    if N is None:
        print('check all items')
        sample_indices = range(0, len(list_result))
    else:
        N = min(N, len(list_result))
        print(f'check {N} items')
        sample_indices = random.sample(range(0, len(list_result)), N)

    # Prepare items with their list indices
    indexed_sample_list = []
    for idx in sample_indices:
        item = list_result[idx]
        item_with_index = item.copy()
        item_with_index['list_index'] = idx  # Add the original index in list_result
        indexed_sample_list.append(item_with_index)

    # Run checks concurrently
    tasks = [
        check_title_appearance(item, page_list, start_index, model)
        for item in indexed_sample_list
    ]
    results = await asyncio.gather(*tasks)
    
    # Process results
    correct_count = 0
    incorrect_results = []
    for result in results:
        if result['answer'] == 'yes':
            correct_count += 1
        else:
            incorrect_results.append(result)
    
    # Calculate accuracy
    checked_count = len(results)
    accuracy = correct_count / checked_count if checked_count > 0 else 0
    print(f"accuracy: {accuracy*100:.2f}%")
    return accuracy, incorrect_results





################### main process #########################################################
async def meta_processor(page_list, mode=None, toc_content=None, toc_page_list=None, start_index=1, opt=None, logger=None):
    print(mode)
    print(f'start_index: {start_index}')
    
    if mode == 'process_toc_with_page_numbers':
        toc_with_page_number = process_toc_with_page_numbers(toc_content, toc_page_list, page_list, toc_check_page_num=opt.toc_check_page_num, model=opt.model, logger=logger)
    elif mode == 'process_toc_no_page_numbers':
        toc_with_page_number = process_toc_no_page_numbers(toc_content, toc_page_list, page_list, model=opt.model, logger=logger)
    else:
        toc_with_page_number = process_no_toc(page_list, start_index=start_index, model=opt.model, logger=logger)
            
    toc_with_page_number = [item for item in toc_with_page_number if item.get('physical_index') is not None] 
    
    # If we don't have any valid items, create a simple structure with just one node
    if not toc_with_page_number:
        logger.info("No valid TOC items found, creating a basic structure")
        toc_with_page_number = [{
            "structure": "1",
            "title": "Main Content",
            "physical_index": start_index
        }]
        return toc_with_page_number
    
    accuracy, incorrect_results = await verify_toc(page_list, toc_with_page_number, start_index=start_index, model=opt.model)
        
    logger.info({
        'mode': mode,
        'accuracy': accuracy,
        'incorrect_results': incorrect_results
    })
    
    if accuracy == 1.0 and len(incorrect_results) == 0:
        return toc_with_page_number
    if accuracy > 0.6 and len(incorrect_results) > 0:
        toc_with_page_number, incorrect_results = await fix_incorrect_toc_with_retries(toc_with_page_number, page_list, incorrect_results,start_index=start_index, max_attempts=3, model=opt.model, logger=logger)
        return toc_with_page_number
    else:
        if mode == 'process_toc_with_page_numbers':
            return await meta_processor(page_list, mode='process_toc_no_page_numbers', toc_content=toc_content, toc_page_list=toc_page_list, start_index=start_index, opt=opt, logger=logger)
        elif mode == 'process_toc_no_page_numbers':
            return await meta_processor(page_list, mode='process_no_toc', start_index=start_index, opt=opt, logger=logger)
        else:
            # Instead of failing, return a basic structure with what we have
            logger.info("Processing accuracy too low, using best available structure")
            # If we have at least some items, use them even with low accuracy
            if toc_with_page_number:
                return toc_with_page_number
            # Last resort fallback - create a simple structure
            return [{
                "structure": "1",
                "title": "Main Content",
                "physical_index": start_index
            }]
        
 
async def process_large_node_recursively(node, page_list, opt=None, logger=None):
    node_page_list = page_list[node['start_index']-1:node['end_index']]
    token_num = sum([page[1] for page in node_page_list])
    
    if node['end_index'] - node['start_index'] > opt.max_page_num_each_node and token_num >= opt.max_token_num_each_node:
        print('large node:', node['title'], 'start_index:', node['start_index'], 'end_index:', node['end_index'], 'token_num:', token_num)

        node_toc_tree = await meta_processor(node_page_list, mode='process_no_toc', start_index=node['start_index'], opt=opt, logger=logger)
        node_toc_tree = await check_title_appearance_in_start_concurrent(node_toc_tree, page_list, model=opt.model, logger=logger)
        
        if node['title'].strip() == node_toc_tree[0]['title'].strip():
            node['nodes'] = post_processing(node_toc_tree[1:], node['end_index'])
            node['end_index'] = node_toc_tree[1]['start_index']
        else:
            node['nodes'] = post_processing(node_toc_tree, node['end_index'])
            node['end_index'] = node_toc_tree[0]['start_index']
        
    if 'nodes' in node and node['nodes']:
        tasks = [
            process_large_node_recursively(child_node, page_list, opt, logger=logger)
            for child_node in node['nodes']
        ]
        await asyncio.gather(*tasks)
    
    return node

async def tree_parser(page_list, opt, doc=None, logger=None):
    check_toc_result = check_toc(page_list, opt)
    logger.info(check_toc_result)

    if check_toc_result.get("toc_content") and check_toc_result["toc_content"].strip() and check_toc_result["page_index_given_in_toc"] == "yes":
        toc_with_page_number = await meta_processor(
            page_list, 
            mode='process_toc_with_page_numbers', 
            start_index=1, 
            toc_content=check_toc_result['toc_content'], 
            toc_page_list=check_toc_result['toc_page_list'], 
            opt=opt,
            logger=logger)
    else:
        toc_with_page_number = await meta_processor(
            page_list, 
            mode='process_no_toc', 
            start_index=1, 
            opt=opt,
            logger=logger)

    toc_with_page_number = add_preface_if_needed(toc_with_page_number)
    toc_with_page_number = await check_title_appearance_in_start_concurrent(
        toc_with_page_number, 
        page_list, 
        model=opt.model, 
        logger=logger
    )
    
    toc_tree = post_processing(toc_with_page_number, len(page_list))
    
    tasks = [
        process_large_node_recursively(node, page_list, opt, logger=logger)
        for node in toc_tree
    ]
    await asyncio.gather(*tasks)
    
    return toc_tree


def page_index_main(doc, opt=None):
    """
    Unified entry point for document structure extraction with specialized
    financial document processing.
    
    This function implements a streamlined workflow for processing financial documents,
    particularly SEC 10-K filings, but works well with other document types too.
    The processing pipeline includes:
    
    1. Document parsing and tokenization
    2. Financial document detection and classification
    3. Table of contents extraction and enhancement
    4. Structure extraction with specialized financial features
    5. Regulatory section identification and validation
    6. Financial table extraction and analysis
    7. Footnote processing and reference linking
    8. Optional enrichment (summaries, IDs, text extraction)
    
    Args:
        doc: PDF document (path or BytesIO)
        opt: Processing options
        
    Returns:
        Dictionary containing document structure and metadata
    """
    logger = JsonLogger(doc)
    
    # Validate input
    is_valid_pdf = (
        (isinstance(doc, str) and os.path.isfile(doc) and doc.lower().endswith(".pdf")) or 
        isinstance(doc, BytesIO)
    )
    if not is_valid_pdf:
        raise ValueError("Unsupported input type. Expected a PDF file path or BytesIO object.")

    logger.info("Starting unified financial document processing pipeline")
    
    # Step 1: Parse document into pages and tokens
    print('Parsing PDF document...')
    page_list = get_page_tokens(doc)

    # Log document statistics
    total_pages = len(page_list)
    total_tokens = sum([page[1] for page in page_list])
    logger.info({'total_page_number': total_pages, 'total_token': total_tokens})
    
    # Step 2: Process document through unified financial document parser
    # This pipeline specializes in SEC 10-K documents but works for all document types
    print(f'Processing document with specialized financial features ({total_pages} pages)...')
    structure = asyncio.run(financial_document_parser(page_list, opt, doc=doc, logger=logger))
    
    # Step 3: Apply requested document enrichments
    
    # Add node IDs if requested
    if getattr(opt, 'if_add_node_id', 'yes') == 'yes':
        print('Adding node IDs to structure...')
        write_node_id(structure)    
    
    # Add summaries if requested
    if getattr(opt, 'if_add_node_summary', 'yes') == 'yes':
        print('Generating summaries for document sections...')
        add_node_text(structure, page_list)
        asyncio.run(generate_summaries_for_structure(structure, model=opt.model))
        remove_structure_text(structure)
        
        # Add node text with labels if requested
        if getattr(opt, 'if_add_node_text', 'yes') == 'yes':
            print('Adding labeled text to document sections...')
            add_node_text_with_labels(structure, page_list)
    
    # Step 4: Create and return final document structure
    
    # Check for financial document indicators in the structure
    is_financial_document = False
    document_type = "Unknown"
    financial_indicators = []
    
    # Extract financial metadata from structure if available
    if structure and isinstance(structure, list) and len(structure) > 0:
        # Check for financial metadata in the top node
        top_node = structure[0]
        if isinstance(top_node, dict) and top_node.get('metadata'):
            metadata = top_node.get('metadata', {})
            document_type = metadata.get('document_type', 'Unknown')
            financial_indicators = metadata.get('financial_indicators', [])
            is_financial_document = bool(document_type != 'Unknown' and document_type != 'Non-SEC Document')
    
    # Add document description if requested
    result = {
        'doc_name': get_pdf_name(doc),
        'is_financial_document': is_financial_document,
        'document_type': document_type,
        'structure': structure,
    }
    
    if getattr(opt, 'if_add_doc_description', 'yes') == 'yes':
        print('Generating document description...')
        doc_description = generate_doc_description(structure, model=opt.model)
        result['doc_description'] = doc_description
        
    logger.info("Document processing complete")
    return result


def page_index(doc, model=None, toc_check_page_num=None, max_page_num_each_node=None, max_token_num_each_node=None,
               if_add_node_id=None, if_add_node_summary=None, if_add_doc_description=None, if_add_node_text=None,
               process_financial_features=None, extract_tables=None, extract_footnotes=None):
    
    user_opt = {
        arg: value for arg, value in locals().items()
        if arg != "doc" and value is not None
    }
    opt = ConfigLoader().load(user_opt)
    return page_index_main(doc, opt)


################### Financial Document Processing #########################################################

async def check_if_financial_document(page_list, model=None, logger=None):
    """
    Enhanced detection of financial documents (SEC 10K) with improved signal recognition.
    
    Args:
        page_list: List of pages with their content
        model: LLM model to use
        logger: Logger instance
        
    Returns:
        Dictionary with detection results and confidence score
    """
    # Check first few pages for indicators of a financial document
    sample_text = ""
    for i in range(min(5, len(page_list))):
        sample_text += page_list[i][0] + "\n\n"
    
    prompt = f"""
    You are a document classification expert specialized in SEC filings. Your task is to determine if the given document is an SEC 10-K financial filing.
    
    Document sample:
    {sample_text}
    
    Please analyze the content and determine if this is an SEC 10-K filing. Look for:
    1. Standard SEC filing language and headers (e.g., "FORM 10-K", "Annual Report", "Securities Exchange Act of 1934")
    2. Financial terminology specific to annual reports (e.g., "Consolidated Statements", "Fiscal Year")
    3. Section titles typical of 10-K filings (e.g., "Item 1. Business", "Item 1A. Risk Factors")
    4. References to financial tables, financial statements, or accounting practices
    5. Company-specific financial information and metrics
    6. References to auditors, accounting standards, or SEC regulations
    
    Reply format:
    {{
        "is_financial_document": true/false,
        "confidence": 0.0-1.0,
        "document_type": "10-K", "10-Q", "8-K", "Other SEC Filing", or "Non-SEC Document",
        "financial_indicators": ["list of specific financial indicators found"],
        "reasoning": "brief explanation for your classification"
    }}
    Directly return the final JSON structure. Do not output anything else.
    """
    
    response = await ChatGPT_API_async(model=model, prompt=prompt)
    result = extract_json(response)
    
    if logger:
        logger.info(f"Financial document detection: {result}")
    
    return result

async def financial_document_parser(page_list, opt, doc=None, logger=None):
    """
    Unified parser for financial documents (SEC 10K) with specialized features.
    
    This function implements a streamlined workflow optimized for financial documents:
    1. Detects financial document features and structure
    2. Extracts TOC with specialized financial section recognition
    3. Processes financial tables, footnotes, and regulatory sections
    4. Validates document completeness according to SEC requirements
    5. Builds a comprehensive tree structure with financial metadata
    
    Args:
        page_list: List of pages with their content
        opt: Options for processing
        doc: Document reference
        logger: Logger instance
        
    Returns:
        Enhanced document structure with financial-specific features
    """
    if logger:
        logger.info("Starting unified financial document parsing")
    
    # Step 1: Perform enhanced financial document detection
    financial_detection = await check_if_financial_document(page_list, model=opt.model, logger=logger)
    is_financial = financial_detection.get("is_financial_document", False)
    doc_type = financial_detection.get("document_type", "Unknown")
    
    if logger:
        logger.info(f"Document classified as: {doc_type} with confidence {financial_detection.get('confidence', 0)}")
    
    # Step 2: Extract and enhance table of contents
    check_toc_result = check_toc(page_list, opt)
    if logger:
        logger.info(f"TOC detection: {check_toc_result['page_index_given_in_toc']}")
    
    # Process based on document characteristics
    if check_toc_result.get("toc_content") and check_toc_result["toc_content"].strip():
        # For financial documents, enhance the TOC with financial section recognition
        if is_financial:
            enhanced_toc = await enhance_toc_for_financial_document(
                check_toc_result["toc_content"],
                model=opt.model
            )
            if logger:
                logger.info(f"Enhanced TOC with {len(enhanced_toc.get('enhanced_toc', []))} sections")
                logger.info(f"Identified {len(enhanced_toc.get('missing_sections', []))} missing sections")
            
            # Use the enhanced TOC content if available
            toc_content = check_toc_result["toc_content"]
        else:
            toc_content = check_toc_result["toc_content"]
        
        # Process with appropriate method based on whether page numbers are present
        if check_toc_result["page_index_given_in_toc"] == "yes":
            toc_with_page_number = await meta_processor(
                page_list, 
                mode='process_toc_with_page_numbers', 
                start_index=1, 
                toc_content=toc_content, 
                toc_page_list=check_toc_result['toc_page_list'], 
                opt=opt,
                logger=logger
            )
        else:
            toc_with_page_number = await meta_processor(
                page_list, 
                mode='process_toc_no_page_numbers', 
                start_index=1, 
                toc_content=toc_content, 
                toc_page_list=check_toc_result['toc_page_list'], 
                opt=opt,
                logger=logger
            )
    else:
        # No TOC found, use direct structure extraction approach
        toc_with_page_number = await meta_processor(
            page_list, 
            mode='process_no_toc', 
            start_index=1, 
            opt=opt,
            logger=logger
        )
    
    # Step 3: Post-process and enhance the structure
    toc_with_page_number = add_preface_if_needed(toc_with_page_number)
    toc_with_page_number = await check_title_appearance_in_start_concurrent(
        toc_with_page_number, 
        page_list, 
        model=opt.model, 
        logger=logger
    )
    
    # Step 4: Transform into a tree structure
    toc_tree = post_processing(toc_with_page_number, len(page_list))
    
    # Step 5: Process large nodes recursively
    tasks = [
        process_large_node_recursively(node, page_list, opt, logger=logger)
        for node in toc_tree
    ]
    await asyncio.gather(*tasks)
    
    # Step 6: Process financial-specific features
    # Always process financial features for identified financial documents
    if is_financial or getattr(opt, 'process_financial_features', 'yes') == 'yes':
        if logger:
            logger.info("Processing financial features for document structure")
        
        # Add document-level financial metadata
        for node in toc_tree:
            if 'metadata' not in node:
                node['metadata'] = {}
            node['metadata']['document_type'] = doc_type
            node['metadata']['financial_indicators'] = financial_detection.get('financial_indicators', [])
        
        # Process financial features for each section
        await process_financial_features_for_structure(toc_tree, page_list, opt, logger)
        
        # Validate regulatory completeness against SEC requirements
        if logger:
            logger.info("Validating regulatory completeness")
        completeness_result = await validate_regulatory_completeness(toc_tree, model=opt.model)
        if logger:
            logger.info(f"Regulatory validation: {completeness_result.get('completeness_score', 0)}")
        
        # Add completeness information to the top-level structure
        for node in toc_tree:
            if 'metadata' not in node:
                node['metadata'] = {}
            node['metadata']['regulatory_validation'] = completeness_result
    
    return toc_tree

async def process_financial_features_for_structure(structure, page_list, opt, logger=None):
    """
    Enhanced processor for financial features in document sections.
    
    This function extracts and processes specialized financial information from each section:
    1. Identifies regulatory sections (e.g., Item 1, Item 1A) per SEC requirements
    2. Extracts and structures financial tables (balance sheets, income statements)
    3. Processes footnotes and builds reference graphs between sections
    4. Identifies financial terms and their definitions
    5. Adds comprehensive financial metadata to each node
    
    Args:
        structure: Document structure tree
        page_list: List of pages with their content
        opt: Options for processing
        logger: Logger instance
        
    Returns:
        Enhanced structure with financial features and metadata
    """
    if logger:
        logger.info("Processing financial features for document structure")
    
    # Track overall financial metrics at document level
    document_metrics = {
        'total_tables': 0,
        'total_footnotes': 0,
        'regulatory_sections': [],
        'key_financial_terms': {},
        'financial_years': set()
    }
    
    async def process_node(node):
        if not isinstance(node, dict):
            return
            
        # Get page range for this node
        start_page = node.get('start_index', 1)
        end_page = node.get('end_index', start_page + 1)
        
        # Extract text for this section
        section_pages = page_list[max(0, start_page-1):max(0, end_page-1)]
        if not section_pages:
            if logger:
                logger.warning(f"No pages found for section {node.get('title', 'Untitled')} - range: {start_page}-{end_page}")
            return
            
        section_text = "\n\n".join([page[0] for page in section_pages])
        
        # Initialize metadata if not present
        if 'metadata' not in node:
            node['metadata'] = {}
        
        # Add section size metadata
        node['metadata']['section_stats'] = {
            'page_count': end_page - start_page,
            'text_length': len(section_text),
            'estimated_tokens': sum([page[1] for page in section_pages]) if all(len(page) > 1 for page in section_pages) else None
        }
        
        # Step 1: Identify if this is a regulatory section
        section_title = node.get('title', '').strip()
        if logger:
            logger.info(f"Processing section: {section_title}")
            
        regulatory_info = await identify_regulatory_section(
            section_title, 
            section_text, 
            model=opt.model
        )
        node['metadata']['regulatory_info'] = regulatory_info
        
        # Track regulatory sections at document level
        if regulatory_info.get('is_regulatory_section', False):
            document_metrics['regulatory_sections'].append({
                'title': section_title,
                'type': regulatory_info.get('section_type'),
                'id': regulatory_info.get('section_id')
            })
        
        # Step 2: Process financial tables based on section type
        # Prioritize processing for financial statement sections
        is_financial_statement = any(term in section_title.lower() for term in [
            'balance sheet', 'income statement', 'statement of operations',
            'cash flow', 'financial statement', 'statement of position',
            'consolidated', 'statements', 'notes', 'financial data'
        ])
        
        priority_processing = regulatory_info.get('is_regulatory_section', False) or is_financial_statement
        
        # Process tables with different strategies based on section type
        if getattr(opt, 'extract_tables', 'yes') == 'yes':
            if priority_processing:
                if logger:
                    logger.info(f"Priority table processing for section: {section_title}")
            
            tables_detected = await detect_financial_tables(section_text, model=opt.model)
            
            if tables_detected.get('tables_detected', False):
                node['metadata']['tables'] = []
                document_metrics['total_tables'] += len(tables_detected.get('tables', []))
                
                for table_info in tables_detected.get('tables', []):
                    # Extract table text based on positions
                    try:
                        start_pos = int(table_info.get('start_position', 0))
                        end_pos = min(int(table_info.get('end_position', len(section_text))), len(section_text))
                        table_text = section_text[start_pos:end_pos]
                        
                        # Parse table structure with appropriate model based on importance
                        table_type = table_info.get('table_type', 'unknown')
                        
                        # Detect fiscal years in table (important for financial analysis)
                        years = re.findall(r'\b(20\d\d|19\d\d)\b', table_text)
                        if years:
                            document_metrics['financial_years'].update(years)
                        
                        # Parse table structure
                        table_structure = await extract_table_structure(
                            table_text, 
                            table_type, 
                            model=opt.model
                        )
                        
                        # Convert to standardized JSON
                        table_json = await table_to_json(table_structure, model=opt.model)
                        
                        # Add to node metadata
                        node['metadata']['tables'].append({
                            'type': table_type,
                            'structure': table_json,
                            'multi_page': table_info.get('multi_page', False),
                            'confidence': table_info.get('confidence', 1.0),
                            'years_mentioned': list(set(years))
                        })
                    except Exception as e:
                        if logger:
                            logger.error(f"Error processing table: {str(e)}")
        
        # Step 3: Process footnotes with reference tracking
        if getattr(opt, 'extract_footnotes', 'yes') == 'yes':
            footnote_refs = await detect_footnote_references(section_text, model=opt.model)
            footnote_contents = await extract_footnote_content(section_text, model=opt.model)
            
            if footnote_refs.get('footnote_references', []) and footnote_contents.get('footnotes', []):
                document_metrics['total_footnotes'] += len(footnote_contents.get('footnotes', []))
                
                reference_graph = await build_reference_graph(
                    footnote_refs, 
                    footnote_contents, 
                    model=opt.model
                )
                node['metadata']['footnotes'] = {
                    'references': footnote_refs.get('footnote_references', []),
                    'contents': footnote_contents.get('footnotes', []),
                    'graph': reference_graph.get('reference_graph', [])
                }
        
        # Step 4: Extract financial terms with stronger financial domain focus
        financial_terms = await extract_financial_terms(section_text, model=opt.model)
        node['metadata']['financial_terms'] = financial_terms.get('financial_terms', [])
        
        # Update document-level financial terms dictionary
        for term in financial_terms.get('financial_terms', []):
            term_name = term.get('term')
            if term_name and term.get('is_explicit_definition', False):
                document_metrics['key_financial_terms'][term_name] = term.get('definition')
        
        # Process child nodes recursively
        if 'nodes' in node and node['nodes']:
            await process_nodes(node['nodes'])
    
    async def process_nodes(nodes):
        tasks = [process_node(node) for node in nodes]
        await asyncio.gather(*tasks)
    
    # Start processing from the top level
    await process_nodes(structure)
    
    # Add document-level metrics to the top nodes
    for node in structure:
        if 'metadata' not in node:
            node['metadata'] = {}
        
        node['metadata']['document_financial_metrics'] = {
            'total_tables': document_metrics['total_tables'],
            'total_footnotes': document_metrics['total_footnotes'],
            'regulatory_sections_count': len(document_metrics['regulatory_sections']),
            'financial_years': list(document_metrics['financial_years']),
            'financial_terms_count': len(document_metrics['key_financial_terms'])
        }
    
    if logger:
        logger.info(f"Processed {document_metrics['total_tables']} financial tables and {document_metrics['total_footnotes']} footnotes")
    
    return structure
