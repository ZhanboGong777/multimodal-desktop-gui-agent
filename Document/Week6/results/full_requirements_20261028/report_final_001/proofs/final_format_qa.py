"""Read-only final format QA; engineering acceptance is bound separately."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import argparse, hashlib, json, re, unicodedata, zipfile
from lxml import etree as E
import pdfplumber
from pypdf import PdfReader

NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
P='http://schemas.openxmlformats.org/package/2006/relationships'
Q=lambda n:'{'+NS['w']+'}'+n
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
norm=lambda s:re.sub(r'\s+','',s)
text=lambda p:''.join(p.xpath('.//w:t/text()',namespaces=NS))

def markdown_lines(raw):
    out=[]
    for line in raw.splitlines():
        if not line.strip() or line.startswith('```'):continue
        if re.fullmatch(r'[| :\-]+',line):continue
        if line.startswith('| '):out.extend(x.strip().replace('`','') for x in line.strip('|').split('|'));continue
        if line.startswith('!['):out.append(re.match(r'!\[([^]]+)\]',line)[1]);continue
        line=re.sub(r'^#+ ','',line)
        line=re.sub(r'\[([^]]+)\]\([^)]+\)',r'\1',line)
        out.append(line.replace('`',''))
    return out

def main(folder):
    docx=next(folder.glob('*.docx'));pdf=next((folder/'render_001').glob('*.pdf'))
    source=Path(__file__).parents[2]/'output/docx/Week6_实验报告_文字修订版_修正.docx'
    with zipfile.ZipFile(docx) as z,zipfile.ZipFile(source) as old:
        root=E.fromstring(z.read('word/document.xml'));body=root.find('w:body',NS)
        same_parts={n:old.read(n)==z.read(n) for n in old.namelist() if n not in ('word/document.xml','word/_rels/document.xml.rels','docProps/core.xml')}
        same_section=E.tostring(body.find('w:sectPr',NS))==E.tostring(E.fromstring(old.read('word/document.xml')).find('w:body/w:sectPr',NS))
        rels={r.get('Id'):r.get('Target') for r in E.fromstring(z.read('word/_rels/document.xml.rels'))}
        links=[{'label':text(n),'target':rels[n.get('{'+NS['r']+'}id')]} for n in body.findall('.//w:hyperlink',NS)]
        blocks=[];headers=[];tables=[]
        for node in body:
            if node.tag==Q('p'):
                if text(node):blocks.append(text(node))
            if node.tag==Q('tbl'):
                rows=[[text(c) for c in row.findall('w:tc',NS)] for row in node.findall('w:tr',NS)]
                headers.append(''.join(rows[0]));tables.append(rows)
                blocks.extend(c for row in rows for c in row)
        images=[]
        for n in body.findall('.//a:blip',NS):
            target=rels[n.get('{'+NS['r']+'}embed')]
            content=z.read('word/'+target)
            images.append({'target':target,'sha256':hashlib.sha256(content).hexdigest()})
        fixed_rows=bool(body.xpath('.//w:trHeight[@w:hRule="exact"]',namespaces=NS))
        repeated_headers=all(t.find('w:tr/w:trPr/w:tblHeader',NS) is not None for t in body.findall('w:tbl',NS))
        direct_sizes=Counter(body.xpath('.//w:rPr/w:sz/@w:val',namespaces=NS))
        direct_east_fonts=Counter(body.xpath('.//w:rPr/w:rFonts/@w:eastAsia',namespaces=NS))
    md=markdown_lines((folder/'report.md').read_text(encoding='utf-8'))
    expected=''.join(blocks)
    page_info=[];actual_chars=[];all_pdf_text=[];header_seen=Counter();excluded=[];pdf_font_sizes=Counter()
    with pdfplumber.open(pdf) as document:
        for page_no,page in enumerate(document.pages,1):
            footer=[c for c in page.chars if c['top']>=735]
            body_chars=[c for c in page.chars if c['top']<735]
            footer_text=''.join(c['text'] for c in sorted(footer,key=lambda x:x['x0']))
            # Repeated table header is removable only when its exact text is
            # found in one line at a page top and the same header preceded it.
            groups=pdfplumber.utils.cluster_objects(body_chars,lambda c:c['top'],3)
            drop_ids=set()
            for chars in groups:
                line=''.join(c['text'] for c in sorted(chars,key=lambda x:x['x0']))
                hit=next((h for h in headers if norm(line)==norm(h)),None)
                if hit is not None:
                    header_seen[hit]+=1
                    if header_seen[hit]>headers.count(hit) and min(c['top'] for c in chars)<110:
                        drop_ids.update(id(c) for c in chars)
                        excluded.append({'page':page_no,'exact_header':hit,'bbox':[min(c['x0'] for c in chars),min(c['top'] for c in chars),max(c['x1'] for c in chars),max(c['bottom'] for c in chars)]})
            kept=[c for c in body_chars if id(c) not in drop_ids]
            actual_chars.extend(c['text'] for c in kept)
            all_pdf_text.append(page.extract_text() or '')
            for c in page.chars:pdf_font_sizes[(c['fontname'],round(c['size'],3))]+=1
            page_info.append({'page':page_no,'width':page.width,'height':page.height,'footer':footer_text,'footer_passed':norm(footer_text)==str(page_no),'body_bottom':max(c['bottom'] for c in body_chars),'footer_top':min(c['top'] for c in footer),'gap_pt':min(c['top'] for c in footer)-max(c['bottom'] for c in body_chars),'png_sha256':sha(folder/'render_001'/f'page-{page_no}.png')})
    actual=''.join(actual_chars)
    expected_counter=Counter(norm(expected));actual_counter=Counter(norm(actual))
    extra=actual_counter-expected_counter;missing=expected_counter-actual_counter
    reader=PdfReader(pdf);uris=[]
    for page_no,page in enumerate(reader.pages,1):
        for ref in page.get('/Annots',[]):
            a=ref.get_object();action=a.get('/A',{})
            if action.get('/URI'):uris.append({'page':page_no,'target':str(action['/URI']),'rect':[float(x) for x in a['/Rect']]})
    pdf_digits=Counter(re.findall(r'\d',norm(actual)));docx_digits=Counter(re.findall(r'\d',norm(expected)))
    code=[];inside=False
    for line in (folder/'report.md').read_text(encoding='utf-8').splitlines():
        if line.startswith('```'):inside=not inside;continue
        if inside:code.append({'text':line,'present':norm(line) in norm('\n'.join(all_pdf_text))})
    checks={'markdown_docx_blocks_exact':blocks==md,'preserved_package_parts_exact':all(same_parts.values()),'section_exact':same_section,'title_final_mark':any('完整验收版' in b for b in blocks[:2]) and not any('待验收稿' in b for b in blocks[:2]),'no_cover_date_line':not any(re.search(r'2026|UTC|Sydney',b) for b in blocks[:2]),'table_header_repetition':repeated_headers,'no_exact_row_heights':not fixed_rows,'pdf_docx_body_characters_exact_after_documented_repeat_headers':not extra and not missing,'pdf_docx_digits_exact':pdf_digits==docx_digits,'pdf_no_replacement_character':'\ufffd' not in actual,'all_footer_numbers_correct':all(p['footer_passed'] for p in page_info),'body_footer_gap_at_least_12pt':all(p['gap_pt']>=12 for p in page_info),'three_live_docx_links':len(links)==3,'pdf_link_target_set_exact':{a['target'] for a in uris}=={a['target'] for a in links},'pdf_link_labels_present':all(norm(a['label']) in norm('\n'.join(all_pdf_text)) for a in links),'all_code_lines_present':all(c['present'] for c in code),'one_visible_image':len(images)==1,'figure_bytes_exact':len(images)==1 and images[0]['sha256']==sha(folder.parent/'assets/perception_warm_latency.png')}
    result={'schema':'week6.report.final_format_qa.v1','recorded_utc':datetime.now(timezone.utc).isoformat(),'meaning':'Final DOCX/PDF layout and content consistency checks only; engineering, actual page viewing and distribution receipts are separate required evidence.','docx':str(docx),'docx_sha256':sha(docx),'pdf':str(pdf),'pdf_sha256':sha(pdf),'markdown_sha256':sha(folder/'report.md'),'checks':checks,'all_checks_passed':all(checks.values()),'page_count':len(page_info),'pages':page_info,'table_count':len(tables),'tables':tables,'docx_links':links,'pdf_uri_annotations':uris,'code_lines':code,'images':images,'raw_pdf_extra_characters':dict(extra),'raw_pdf_missing_characters':dict(missing),'documented_repeated_header_exclusions':excluded,'docx_direct_sizes_half_points':dict(direct_sizes),'docx_east_asian_fonts':dict(direct_east_fonts),'pdf_font_sizes':[{'font':k[0],'pt':k[1],'characters':v} for k,v in pdf_font_sizes.items()]}
    output=folder/'final_format_qa_001.json'
    with output.open('xb') as f:f.write((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    (folder/'final_pdf_text_001.txt').write_bytes('\n\n'.join(all_pdf_text).encode('utf-8'))
    print(json.dumps({'output':str(output),'sha256':sha(output),'checks':checks,'pages':len(page_info),'extra':dict(extra),'missing':dict(missing)},ensure_ascii=False,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path)
    main(parser.parse_args().folder)
