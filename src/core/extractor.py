import easyocr
import re
import difflib
import logging
import spacy
import numpy as np
from typing import Dict, List, Any, Optional
from src.utils.image_utils import load_image_from_path, convert_pdf_to_images, preprocess_image_for_ocr
from src.core.schema import ExtractionResult, ExtractedField

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class DocumentExtractor:
    def __init__(self, use_gpu: bool = False):
        import torch
        logger.info(f"Initializing EasyOCR Reader (GPU={use_gpu})...")

        original_load = torch.load

        def safe_load(*args, **kwargs):
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return original_load(*args, **kwargs)

        try:
            torch.load = safe_load
            self.reader = easyocr.Reader(['en'], gpu=use_gpu)
        finally:
            torch.load = original_load

        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    # --------------------------------------------------
    # MAIN ENTRY
    # --------------------------------------------------
    def extract_from_file(self, file_path: str) -> Dict[str, Any]:
        try:
            images = (
                convert_pdf_to_images(file_path)
                if file_path.lower().endswith(".pdf")
                else [load_image_from_path(file_path)]
            )

            all_fields: Dict[str, Dict] = {}
            detected_type = "UNKNOWN"
            full_text = ""

            for img in images:
                processed_img = preprocess_image_for_ocr(img)
                ocr = self.reader.readtext(processed_img, detail=1)
                lines_text = "\n".join([r[1] for r in ocr])
                full_text += lines_text + "\n"

                if detected_type == "UNKNOWN":
                    detected_type = self._identify_document_type(lines_text)

                page_fields = self._analyze_page(ocr, lines_text, detected_type)

                # 🔥 FIX: confidence-aware merge
                for k, v in page_fields.items():
                    if k not in all_fields or v["confidence"] > all_fields[k]["confidence"]:
                        all_fields[k] = v

            if detected_type == "UNKNOWN":
                detected_type = self._identify_document_type(full_text)

            final_fields = {
                k: ExtractedField(value=v["value"], confidence=v["confidence"])
                for k, v in all_fields.items()
                if v.get("value")
            }

            return ExtractionResult(
                document_type=detected_type,
                status="success",
                fields=final_fields
            ).model_dump(exclude_none=True)

        except Exception as e:
            logger.error("Extraction failed", exc_info=True)
            return ExtractionResult(
                document_type="UNKNOWN",
                status="failure",
                message=str(e)
            ).model_dump()

    # --------------------------------------------------
    # PAGE ANALYSIS
    # --------------------------------------------------
    def _analyze_page(self, ocr, text: str, doc_type: str) -> Dict[str, Dict]:
        fields = {}
        lines = self._group_by_lines(ocr)

        date = self._find_pattern(text, r'\d{2}[/-]\d{2}[/-]\d{4}', 0.90)

        if doc_type == "PAN":
            # PAN Number
            pan = self._find_pan(text)
            if pan:
                fields["pan_number"] = pan
            else:
                anchor_pan = self._extract_pan_by_anchor(lines)
                if anchor_pan:
                    fields["pan_number"] = anchor_pan

            # Name
            name = self._extract_pan_name(lines, text)
            if name:
                fields["name"] = name

            # DOB
            if date:
                fields["date_of_birth"] = date

        elif doc_type == "GST":
            # GSTIN
            gstin = self._find_pattern(text, r'\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]', 0.98)
            if not gstin:
                gstin = self._find_gstin_by_anchor(lines)
            if gstin:
                fields["gstin"] = gstin

            # Legal Name
            legal = self._extract_gst_field(lines, ["LEGAL", "NAME"], ["TRADE", "ADDRESS", "CONSTITUTION"])
            if not legal:
                legal = self._extract_gst_field(lines, ["TRADE", "NAME"], ["ADDRESS", "CONSTITUTION"])
            if legal:
                fields["legal_name"] = legal

            # Address
            addr = self._extract_gst_field(lines, ["ADDRESS"], ["DATE", "CONSTITUTION"])
            if not addr:
                 addr = self._extract_gst_field(lines, ["PRINCIPAL", "PLACE"], ["DATE", "CONSTITUTION", "NATURE"])
            if addr:
                fields["address"] = addr

            # Registration Date
            if date:
                fields["registration_date"] = date

            # Registration Type
            reg_type = self._extract_gst_field(lines, ["REGISTRATION", "TYPE"], ["DATE", "ADDRESS", "LEGAL"])
            if reg_type:
                fields["registration_type"] = reg_type

        elif doc_type == "CHEQUE":
            # Account Holder / Payee
            payee = self._extract_cheque_payee(lines, text)
            if payee:
                fields["payee_name"] = payee

            # Account Number
            acc = self._extract_cheque_acc_no(text, lines)
            if acc:
                fields["account_number"] = acc

            # IFSC - Only for CHEQUE/BANK documents
            ifsc = self._extract_ifsc(lines, text)
            if ifsc:
                fields["ifsc_code"] = ifsc

        return fields

    def _extract_ifsc(self, lines, text):
        """Extract IFSC code - only for bank documents."""
        ifsc = None
        
        # 1. Anchor-based (HIGHEST PRIORITY)
        ifsc_idx = self._find_anchor_line_index(lines, ["IFSC", "CODE"], True)
        if ifsc_idx == -1: 
            ifsc_idx = self._find_anchor_line_index(lines, ["IFSC"], True)
        
        if ifsc_idx != -1:
            # Check anchor line + next line + previous line
            check_indices = [ifsc_idx]
            if ifsc_idx + 1 < len(lines): check_indices.append(ifsc_idx + 1)
            if ifsc_idx - 1 >= 0: check_indices.append(ifsc_idx - 1)
            
            for i in check_indices:
                line_text = lines[i]["text"].upper()
                line_text = re.sub(r'\bIFSC\b', '', line_text, flags=re.I)
                line_text = re.sub(r'\bCODE\b', '', line_text, flags=re.I)
                clean_line = re.sub(r'[^A-Z0-9]', '', line_text)
                
                # Try strict match
                loc_match = re.search(r'[A-Z]{4}0[A-Z0-9]{6}', clean_line)
                if loc_match:
                    ifsc = {"value": loc_match.group(0), "confidence": lines[i]["confidence"]}
                    break
                
                # Try repair on 11-char candidates
                for match in re.finditer(r'[A-Z0-9]{11}', clean_line):
                    candidate = match.group(0)
                    repaired = self._repair_ifsc_candidate(candidate)
                    if repaired:
                        ifsc = {"value": repaired, "confidence": lines[i]["confidence"] * 0.9}
                        break
                if ifsc: break
        
        # 2. Direct pattern match (only if anchor found IFSC keyword in document)
        if not ifsc and ifsc_idx != -1:
            ifsc = self._find_pattern(text, r'[A-Z]{4}0[A-Z0-9]{6}', 0.95)
            
        return ifsc

    # ... [Helpers] ...

    def _extract_cheque_payee(self, lines, text):
        # 0. Helper to clean titles
        def clean_name(val):
            val = re.sub(r'^(MR\.?|MRS\.?|MS\.?|M/S\.?|SHRI\.?|SMT\.?)\s*', '', val.strip(), flags=re.I).strip()
            return val

        # 1. Key-Value Pairs (HIGHEST PRIORITY for structured certificates)
        # Format might be: "Account Holder Name : {name}" OR "Account Holder Name {name}" OR separate lines
        for anchors in [["HOLDER", "NAME"], ["ACCOUNT", "HOLDER"], ["BENEFICIARY", "NAME"], ["CUSTOMER", "NAME"], ["PAY"]]:
            idx = self._find_anchor_line_index(lines, anchors, True)
            if idx != -1:
                txt = lines[idx]["text"]
                val = None
                
                # Try same-line extraction with delimiter
                if ":" in txt: 
                    val = txt.split(":", 1)[1].strip()
                elif "-" in txt and "HOLDER" not in txt.split("-")[0].upper(): 
                    # Avoid splitting "ACCOUNT-HOLDER"
                    val = txt.split("-", 1)[1].strip()
                elif "PAY" in anchors: 
                    val = re.sub(r'PAY', '', txt, flags=re.I).strip()
                else:
                    # Try stripping anchor keywords (space-separated format)
                    # e.g., "Account Holder Name RANVEER RANA" -> "RANVEER RANA"
                    remainder = txt.upper()
                    for kw in anchors:
                        remainder = re.sub(r'\b' + kw + r'\b', '', remainder, flags=re.I)
                    # Also remove common prefixes like "Account"
                    remainder = re.sub(r'\bACCOUNT\b', '', remainder, flags=re.I)
                    remainder = remainder.strip()
                    if len(remainder) > 2 and not re.search(r'\d', remainder):
                        val = remainder
                
                if val:
                    val = clean_name(val)
                    if len(val) > 2 and "..." not in val:
                         return {"value": val, "confidence": lines[idx]["confidence"]}
                
                # If no inline value, check next line
                if idx + 1 < len(lines):
                    nxt = clean_name(lines[idx+1]["text"])
                    if len(nxt) > 2 and not re.search(r'\d', nxt) and "..." not in nxt:
                        return {"value": nxt, "confidence": lines[idx+1]["confidence"]}
            
        # 2. Signatory (Bottom-Up)
        sig_idx = self._find_anchor_line_index(lines, ["SIGNATORY"], True)
        if sig_idx == -1: sig_idx = self._find_anchor_line_index(lines, ["AUTHORISED"], True)
        
        if sig_idx != -1:
            for offset in [1, 2]:
                if sig_idx - offset >= 0:
                    val = clean_name(lines[sig_idx - offset]["text"])
                    if len(val) > 3 and not re.search(r'\d', val):
                        if "FOR " not in val.upper() and "YOURS" not in val.upper():
                             return {"value": val, "confidence": lines[sig_idx - offset]["confidence"]}

        # 3. Relative Name (S/O)
        for line in lines:
            txt = line["text"]
            m = re.search(r'([A-Za-z\.\s]+)\s+(S/O|D/O|W/O)\s+', txt, re.I)
            if m:
                val = clean_name(m.group(1))
                if len(val) > 3 and not re.search(r'\d', val):
                     return {"value": val, "confidence": line["confidence"]}

        # 4. Anchor-based
        for anchors in [["CERTIFY", "THAT"], ["ACCOUNT", "OF"], ["REQUEST", "OF"]]:
            idx = self._find_anchor_line_index(lines, anchors, True)
            if idx != -1:
                line_txt = lines[idx]["text"]
                key = anchors[-1]
                parts = re.split(key, line_txt, flags=re.I)
                if len(parts) > 1:
                     val = clean_name(parts[-1])
                     if len(val) > 3 and not re.search(r'\d', val):
                         return {"value": val, "confidence": lines[idx]["confidence"]}
                
                if idx + 1 < len(lines):
                    val = clean_name(lines[idx+1]["text"])
                    if len(val) > 3 and not re.search(r'\d', val):
                         return {"value": val, "confidence": lines[idx+1]["confidence"]}
        
        # 5. Salutation Scan
        for line in lines:
            txt = line["text"].strip()
            if re.match(r'^(MR\.|MRS\.|MS\.|SHRI|SMT)\s+[A-Z\s]+$', txt, re.I):
                val = clean_name(txt)
                if len(val) > 3 and not re.search(r'\d', val):
                     if "MANAGER" not in val.upper():
                         return {"value": val, "confidence": line["confidence"]}

        # 6. Spacy
        if "CERTIFY" in text.upper() or "BANK" in text.upper():
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    t = ent.text.strip()
                    if len(t) < 4: continue
                    if any(x in t.upper() for x in ["MANAGER", "OFFICER", "SIGNATORY", "BANK", "DATE", "PLACE"]): continue
                    return {"value": t, "confidence": 0.70}

        return None

    def _clean_title(self, val):
        return re.sub(r'^(MR\.?|MRS\.?|MS\.?|M/S\.?|SHRI\.?|SMT\.?)\s*', '', val.strip(), flags=re.I).strip()

    # ... [Existing Methods] ...

    def _repair_ifsc_candidate(self, word: str) -> Optional[str]:
        # IFSC Format: 4 Letters + '0' + 6 Alphanumeric
        chars = list(word)
        
        # 1. Bank Code (4 Letters)
        for i in range(4):
            if chars[i].isdigit(): chars[i] = self._digit_to_char(chars[i])
            
        # 2. Fifth char must be '0'
        if chars[4] != '0':
            # Common OCR errors for 0: O, D, Q
            if chars[4] in ['O', 'D', 'Q', 'U']: chars[4] = '0'
            elif chars[4].isalpha(): chars[4] = '0' # Force 0 if letter
            
        # 3. Branch Code (6 Alphanum) - No strict repair possible without list
        
        repaired = "".join(chars)
        if re.match(r'[A-Z]{4}0[A-Z0-9]{6}', repaired):
            return repaired
        return None

    def _digit_to_char(self, c): 
        return {'0':'O','1':'I','5':'S','8':'B','2':'Z','6':'G','4':'A', '7':'Z', '3':'E'}.get(c,c)
        
    def _char_to_digit(self, c): 
        return {'O':'0','I':'1','L':'1','S':'5','B':'8','Z':'2','G':'6','A':'4', 'D':'0', 'Q':'0'}.get(c,c)

    def _extract_pan_by_anchor(self, lines):
        """Extract PAN by looking for PAN label anchor."""
        idx = self._find_anchor_line_index(lines, ["PAN"], True)
        if idx != -1:
            # Check same line for PAN pattern
            txt = re.sub(r'[^A-Z0-9]', '', lines[idx]["text"].upper())
            m = re.search(r'[A-Z]{5}[0-9]{4}[A-Z]', txt)
            if m:
                return {"value": m.group(0), "confidence": lines[idx]["confidence"]}
            
            # Check next line
            if idx + 1 < len(lines):
                nxt = re.sub(r'[^A-Z0-9]', '', lines[idx + 1]["text"].upper())
                m = re.search(r'[A-Z]{5}[0-9]{4}[A-Z]', nxt)
                if m:
                    return {"value": m.group(0), "confidence": lines[idx + 1]["confidence"]}
        return None

    # --------------------------------------------------
    # HELPERS
    # --------------------------------------------------
    def _find_pan(self, text: str) -> Optional[Dict]:
        match = re.search(r'[A-Z]{5}[0-9]{4}[A-Z]', text)
        if match:
            return {"value": match.group(0), "confidence": 0.98}

        spaced = re.search(r'[A-Z]\s*[A-Z]\s*[A-Z]\s*[A-Z]\s*[A-Z]\s*\d\s*\d\s*\d\s*\d\s*[A-Z]', text)
        if spaced:
            val = spaced.group(0).replace(" ", "")
            return {"value": val, "confidence": 0.90}

        for w in text.split():
            c = re.sub(r'[^A-Z0-9]', '', w.upper())
            if len(c) == 10:
                repaired = self._repair_pan_candidate(c)
                if repaired:
                    return {"value": repaired, "confidence": 0.80}
        return None

    def _repair_pan_candidate(self, word: str) -> Optional[str]:
        chars = list(word)
        for i in range(5):
            if chars[i].isdigit():
                chars[i] = self._digit_to_char(chars[i])
        for i in range(5, 9):
            if chars[i].isalpha():
                chars[i] = self._char_to_digit(chars[i])
        if chars[9].isdigit():
            chars[9] = self._digit_to_char(chars[9])

        repaired = "".join(chars)
        return repaired if re.match(r'[A-Z]{5}\d{4}[A-Z]', repaired) else None

    def _extract_pan_name(self, lines, text):
        start = self._find_anchor_line_index(lines, ["INCOME", "TAX"], True)
        if start == -1:
            start = self._find_anchor_line_index(lines, ["GOVT", "INDIA"], True)

        end = self._find_anchor_line_index(lines, ["FATHER", "NAME"], True)

        candidates = []
        
        # Strategy: Populate candidates list based on found anchors
        if start != -1:
            # Search downwards from header
            limit = end if end != -1 else min(start + 5, len(lines))
            for i in range(start + 1, limit):
                candidates.append(lines[i])
        elif end != -1:
            # Search upwards from footer (Father's Name)
            for i in range(max(0, end - 3), end):
                candidates.append(lines[i])

        # Filter and score candidates
        best_candidate = None
        max_score = 0

        for c in candidates:
            t = c["text"].strip()
            
            # Strip common labels like "Name:", "NAME :"
            t = re.sub(r'^(NAME\s*[:\-])?\s*', '', t, flags=re.I).strip()
            
            u = t.upper()
            
            # 1. Hard Constraints
            if len(t) < 3: continue
            if re.search(r'\d', t): continue
            
            # Filter standard headers/keywords
            if any(x in u for x in ["INCOME", "TAX", "INDIA", "GOVT", "PERMANENT", "ACCOUNT", "NUMBER", "CARD"]):
                continue
            
            # 2. Scorable Heuristics
            score = 10
            # Prefer lines with typical name spacing (e.g. "FIRST LAST")
            if " " in t: score += 5
            # penalize very short words
            if len(t) < 4: score -= 2
            # penalize all lowercase (PAN names are usually UPPER)
            if t.islower(): score -= 5
            
            if score > max_score:
                max_score = score
                best_candidate = {"value": t, "confidence": c["confidence"]}

        if best_candidate:
            return best_candidate

        # spaCy fallback ONLY for PAN
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON" and len(ent.text) > 3:
                return {"value": ent.text, "confidence": 0.65}
        return None

    def _identify_document_type(self, text: str) -> str:
        t = text.upper()
        if "INCOME TAX" in t or "PERMANENT ACCOUNT" in t:
            return "PAN"
        if "GSTIN" in t or "REGISTRATION CERTIFICATE" in t:
            return "GST"
        if "CHEQUE" in t or "IFSC" in t:
            return "CHEQUE"
        return "UNKNOWN"

    def _find_pattern(self, text, pattern, confidence):
        m = re.search(pattern, text)
        return {"value": m.group(0), "confidence": confidence} if m else None

    def _find_gstin_by_anchor(self, lines):
        idx = self._find_anchor_line_index(lines, ["GSTIN"], True)
        if idx != -1:
            for i in range(idx, min(idx + 2, len(lines))):
                line_text = lines[i]["text"].upper()
                
                # Strip GSTIN label
                line_text = re.sub(r'\bGSTIN\b', '', line_text, flags=re.I)
                line_text = re.sub(r'[:\-]', '', line_text)
                
                # Extract alphanumeric only
                clean = re.sub(r'[^A-Z0-9]', '', line_text)
                
                # 1. Try strict 15-char match
                if len(clean) >= 15:
                    # Look for pattern in the cleaned text
                    m = re.search(r'\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]', clean)
                    if m:
                        return {"value": m.group(0), "confidence": lines[i]["confidence"]}
                    
                    # Try to extract first 15 chars and repair
                    candidate = clean[:15]
                    repaired = self._repair_gstin_candidate(candidate)
                    if repaired:
                        return {"value": repaired, "confidence": lines[i]["confidence"] * 0.9}
                
                # 2. Try finding any 15-char alphanumeric substring
                for word in re.findall(r'[A-Z0-9]{15,}', clean):
                    if len(word) >= 15:
                        candidate = word[:15]
                        repaired = self._repair_gstin_candidate(candidate)
                        if repaired:
                            return {"value": repaired, "confidence": lines[i]["confidence"] * 0.9}
        return None

    def _repair_gstin_candidate(self, word: str) -> Optional[str]:
        chars = list(word)
        # 1. State Code (0-2) -> Digits
        for i in [0, 1]:
            if chars[i].isalpha(): chars[i] = self._char_to_digit(chars[i])
            
        # 2. PAN Part (2-7) -> Letters
        for i in range(2, 7):
            if chars[i].isdigit(): chars[i] = self._digit_to_char(chars[i])
            
        # 3. PAN Part (7-11) -> Digits
        for i in range(7, 11):
            if chars[i].isalpha(): chars[i] = self._char_to_digit(chars[i])
            
        # 4. PAN Last Char (11) -> Letter
        if chars[11].isdigit(): chars[11] = self._digit_to_char(chars[11])

        # 5. Entity Code (12) -> Alphanumeric (keep as is)
        
        # 6. Default 'Z' (13) -> 'Z'
        if chars[13] == '2': chars[13] = 'Z'
            
        repaired = "".join(chars)
        # Validate against user regex pattern
        if re.match(r'\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]', repaired):
             return repaired
        return None

    def _find_anchor_line_index(self, lines, keywords, fuzzy=False):
        for i, line in enumerate(lines):
            t = line["text"].upper()
            hits = 0
            for kw in keywords:
                if kw in t:
                    hits += 1
                elif fuzzy:
                    # Strip punctuation for better fuzzy matching
                    words = [re.sub(r'[^\w]', '', w) for w in t.split()]
                    if difflib.get_close_matches(kw, words, cutoff=0.80):
                        hits += 1
            if hits == len(keywords):
                return i
        return -1

    def _extract_gst_field(self, lines, labels, stop):
        idx = self._find_anchor_line_index(lines, labels, True)
        if idx != -1:
            # Check same line first (e.g. "Legal Name: XXX")
            txt = lines[idx]["text"]
            # removing labels from text to see if value remains
            # simplistically split by colon or just taking end of string if long enough
            if ":" in txt:
                val = txt.split(":", 1)[1].strip()
                if len(val) > 2 and not self._fuzzy_contains(val.upper(), stop):
                    return {"value": val, "confidence": lines[idx]["confidence"]}
            
            # Check next line
            if idx + 1 < len(lines):
                nxt = lines[idx + 1]["text"]
                if not self._fuzzy_contains(nxt.upper(), stop) and not self._fuzzy_contains(nxt.upper(), labels):
                    return {"value": nxt, "confidence": lines[idx + 1]["confidence"]}
        return None



    def _extract_cheque_acc_no(self, text, lines):
        idx = self._find_anchor_line_index(lines, ["A/C", "NO"], True)
        if idx != -1 and idx + 1 < len(lines):
            return {"value": lines[idx + 1]["text"], "confidence": lines[idx + 1]["confidence"]}

        m = re.search(r'\b\d{9,18}\b', text)
        return {"value": m.group(0), "confidence": 0.85} if m else None

    def _group_by_lines(self, ocr, y_thresh=10):
        if not ocr:
            return []
        ocr = sorted(ocr, key=lambda x: (x[0][0][1], x[0][0][0]))
        lines, buf, last_y = [], [], -1

        for box, text, conf in ocr:
            y = box[0][1]
            if last_y == -1 or abs(y - last_y) < y_thresh:
                buf.append((box, text, conf))
            else:
                lines.append(self._merge_line(buf))
                buf = [(box, text, conf)]
            last_y = y

        if buf:
            lines.append(self._merge_line(buf))
        return lines

    def _merge_line(self, items):
        items.sort(key=lambda x: x[0][0][0])
        return {
            "text": " ".join(i[1] for i in items),
            "confidence": sum(i[2] for i in items) / len(items),
            "box": items[0][0]
        }

    def _fuzzy_contains(self, text, words):
        for w in words:
            if w in text:
                return True
        return False

    def _digit_to_char(self, c): return {'0':'O','1':'I','5':'S','8':'B','2':'Z','6':'G','4':'A', '7':'Z'}.get(c,c)
    def _char_to_digit(self, c): return {'O':'0','I':'1','L':'1','S':'5','B':'8','Z':'2','G':'6','A':'4'}.get(c,c)
