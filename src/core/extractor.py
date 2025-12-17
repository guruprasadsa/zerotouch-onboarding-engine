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
            # ML-based Entity Extraction: Add GSTIN pattern
            ruler = self.nlp.add_pipe("entity_ruler", before="ner")
            patterns = [
                {"label": "GSTIN", "pattern": [{"TEXT": {"REGEX": r"\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]"}}]}
            ]
            ruler.add_patterns(patterns)
        except Exception:
            try:
                spacy.cli.download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
            except Exception:
                logger.warning("Spacy model failed to load.")
                self.nlp = None

    # MAIN ENTRY
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
                # Simple grayscale preprocessing for all document types
                processed_img = preprocess_image_for_ocr(img)
                ocr = self.reader.readtext(processed_img, detail=1)
                lines_text = "\n".join([r[1] for r in ocr])
                full_text += lines_text + "\n"

                if detected_type == "UNKNOWN":
                    detected_type = self._identify_document_type(lines_text)

                page_fields = self._analyze_page(ocr, lines_text, detected_type)

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

    # PAGE ANALYSIS
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
            gstin = self._extract_gstin_comprehensive(lines, text, ocr)
            if gstin:
                fields["gstin"] = gstin

            # Legal Name
            legal = self._extract_gst_field(lines, ["LEGAL", "NAME"], ["TRADE", "ADDRESS", "CONSTITUTION"])
            if not legal:
                legal = self._extract_gst_field(lines, ["TRADE", "NAME"], ["ADDRESS", "CONSTITUTION"])
            if not legal:
                 legal = self._extract_gst_field(lines, ["NAME", "OF", "TAXPAYER"], ["ADDRESS"])
            if legal:
                # Post-process to fix common OCR punctuation errors
                val = legal["value"]
                # Fix "Name_ Surname" scan noise (underscore -> hyphen)
                val = val.replace('_', '-')
                # Fix space around commas and semicolons
                val = val.replace(';', ',')
                val = re.sub(r'\s*,\s*', ', ', val)
                # Fix "Name - Surname" space issues (hyphen-space -> hyphen)
                val = re.sub(r'\s*-\s*', '-', val)
                legal["value"] = val
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
            valid_types = ["Regular", "Composition", "Casual Taxable Person", 
                          "SEZ Unit", "SEZ Developer", "Input Service Distributor"]
            reg_type = self._extract_gst_field(lines, ["REGISTRATION", "TYPE"], 
                                             ["DATE", "ADDRESS", "LEGAL"], 
                                             allowed_values=valid_types)
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
        
        # Anchor-based
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
                
                for match in re.finditer(r'[A-Z0-9]{11}', clean_line):
                    candidate = match.group(0)
                    repaired = self._repair_ifsc_candidate(candidate)
                    if repaired:
                        ifsc = {"value": repaired, "confidence": lines[i]["confidence"] * 0.9}
                        break
                if ifsc: break
        
        # Direct pattern match
        if not ifsc and ifsc_idx != -1:
            ifsc = self._find_pattern(text, r'[A-Z]{4}0[A-Z0-9]{6}', 0.95)
            
        return ifsc

    # ... [Helpers] ...

    def _extract_cheque_payee(self, lines, text):
        # Helper to clean titles
        def clean_name(val):
            val = re.sub(r'^(MR\.?|MRS\.?|MS\.?|M/S\.?|SHRI\.?|SMT\.?)\s*', '', val.strip(), flags=re.I).strip()
            return val

        # Key-Value Pairs (HIGHEST PRIORITY for structured certificates)
        for anchors in [["HOLDER", "NAME"], ["ACCOUNT", "HOLDER"], ["BENEFICIARY", "NAME"], ["CUSTOMER", "NAME"], ["PAY"]]:
            idx = self._find_anchor_line_index(lines, anchors, True)
            if idx != -1:
                txt = lines[idx]["text"]
                val = None
                
                if ":" in txt: 
                    val = txt.split(":", 1)[1].strip()
                elif "-" in txt and "HOLDER" not in txt.split("-")[0].upper(): 
                    val = txt.split("-", 1)[1].strip()
                elif "PAY" in anchors: 
                    val = re.sub(r'PAY', '', txt, flags=re.I).strip()
                else:
                    remainder = txt.upper()
                    for kw in anchors:
                        remainder = re.sub(r'\b' + kw + r'\b', '', remainder, flags=re.I)
                    # remove common prefixes like "Account"
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
            
        # Signatory (Bottom-Up)
        sig_idx = self._find_anchor_line_index(lines, ["SIGNATORY"], True)
        if sig_idx == -1: sig_idx = self._find_anchor_line_index(lines, ["AUTHORISED"], True)
        
        if sig_idx != -1:
            for offset in [1, 2]:
                if sig_idx - offset >= 0:
                    val = clean_name(lines[sig_idx - offset]["text"])
                    if len(val) > 3 and not re.search(r'\d', val):
                        if "FOR " not in val.upper() and "YOURS" not in val.upper():
                             return {"value": val, "confidence": lines[sig_idx - offset]["confidence"]}

        # Relative Name (S/O)
        for line in lines:
            txt = line["text"]
            m = re.search(r'([A-Za-z\.\s]+)\s+(S/O|D/O|W/O)\s+', txt, re.I)
            if m:
                val = clean_name(m.group(1))
                if len(val) > 3 and not re.search(r'\d', val):
                     return {"value": val, "confidence": line["confidence"]}

        # Anchor-based
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
        
        # Salutation Scan
        for line in lines:
            txt = line["text"].strip()
            if re.match(r'^(MR\.|MRS\.|MS\.|SHRI|SMT)\s+[A-Z\s]+$', txt, re.I):
                val = clean_name(txt)
                if len(val) > 3 and not re.search(r'\d', val):
                     if "MANAGER" not in val.upper():
                         return {"value": val, "confidence": line["confidence"]}

        # Spacy
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

    # Existing Methods

    def _repair_ifsc_candidate(self, word: str) -> Optional[str]:
        # IFSC Format: 4 Letters + '0' + 6 Alphanumeric
        chars = list(word)
        
        # Bank Code (4 Letters)
        for i in range(4):
            if chars[i].isdigit(): chars[i] = self._digit_to_char(chars[i])
            
        # Fifth char must be '0'
        if chars[4] != '0':
            # Common OCR errors for 0: O, D, Q
            if chars[4] in ['O', 'D', 'Q', 'U']: chars[4] = '0'
            elif chars[4].isalpha(): chars[4] = '0' # Force 0 if letter
            
        # Branch Code (6 Alphanum)
        
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
            
            # Try repair on 10-char candidates
            for start in range(max(0, len(txt) - 10)):
                candidate = txt[start:start+10]
                repaired = self._repair_pan_candidate(candidate)
                if repaired:
                    return {"value": repaired, "confidence": lines[idx]["confidence"] * 0.9}
            
            # Check next line
            if idx + 1 < len(lines):
                nxt = re.sub(r'[^A-Z0-9]', '', lines[idx + 1]["text"].upper())
                m = re.search(r'[A-Z]{5}[0-9]{4}[A-Z]', nxt)
                if m:
                    return {"value": m.group(0), "confidence": lines[idx + 1]["confidence"]}
                
                # Try repair on next line
                for start in range(max(0, len(nxt) - 10)):
                    candidate = nxt[start:start+10]
                    repaired = self._repair_pan_candidate(candidate)
                    if repaired:
                        return {"value": repaired, "confidence": lines[idx + 1]["confidence"] * 0.9}
        return None

    # HELPERS
    def _find_pan(self, text: str) -> Optional[Dict]:
        """Find PAN number using multiple strategies."""
        clean_text = text.upper()
        
        # Strategy 1: Direct regex match (exact format)
        match = re.search(r'[A-Z]{5}[0-9]{4}[A-Z]', clean_text)
        if match:
            return {"value": match.group(0), "confidence": 0.98}

        # Strategy 2: Spaced pattern (OCR may insert spaces)
        spaced = re.search(r'[A-Z]\s*[A-Z]\s*[A-Z]\s*[A-Z]\s*[A-Z]\s*\d\s*\d\s*\d\s*\d\s*[A-Z]', clean_text)
        if spaced:
            val = spaced.group(0).replace(" ", "")
            return {"value": val, "confidence": 0.90}

        # Strategy 3: Try each word - repair if needed
        for w in text.split():
            c = re.sub(r'[^A-Z0-9]', '', w.upper())
            if len(c) == 10:
                repaired = self._repair_pan_candidate(c)
                if repaired:
                    return {"value": repaired, "confidence": 0.85}
        
        # Strategy 4: Try 10-char substrings from each word
        for w in text.split():
            c = re.sub(r'[^A-Z0-9]', '', w.upper())
            if len(c) > 10:
                for start in range(len(c) - 9):
                    candidate = c[start:start+10]
                    repaired = self._repair_pan_candidate(candidate)
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
        """Extract cardholder name from PAN card."""
        
        # Strategy 1: Look for explicit "Name" label (new format)
        name_idx = self._find_anchor_line_index(lines, ["NAME"], True)
        if name_idx != -1:
            # Check if the same line has value after label
            line_text = lines[name_idx]["text"]
            if ":" in line_text:
                val = line_text.split(":", 1)[1].strip()
                if len(val) > 2 and not re.search(r'\d', val):
                    if "FATHER" not in line_text.upper():
                        return {"value": val, "confidence": lines[name_idx]["confidence"]}
            
            # Check next line for name value
            if name_idx + 1 < len(lines):
                # Verify this isn't "Father's Name" line
                if "FATHER" not in lines[name_idx]["text"].upper():
                    next_text = lines[name_idx + 1]["text"].strip()
                    # Make sure next line isn't a label
                    if (len(next_text) > 2 and 
                        not re.search(r'\d', next_text) and
                        "FATHER" not in next_text.upper() and
                        "DATE" not in next_text.upper() and
                        "BIRTH" not in next_text.upper()):
                        return {"value": next_text, "confidence": lines[name_idx + 1]["confidence"]}
        
        # Strategy 2: Geometric anchor-based (traditional format)
        start = self._find_anchor_line_index(lines, ["INCOME", "TAX"], True)
        if start == -1:
            start = self._find_anchor_line_index(lines, ["GOVT", "INDIA"], True)
        if start == -1:
            start = self._find_anchor_line_index(lines, ["PERMANENT", "ACCOUNT"], True)

        end = self._find_anchor_line_index(lines, ["FATHER", "NAME"], True)
        if end == -1:
            end = self._find_anchor_line_index(lines, ["DATE", "BIRTH"], True)

        candidates = []
        
        if start != -1:
            limit = end if end != -1 else min(start + 6, len(lines))
            for i in range(start + 1, limit):
                candidates.append(lines[i])
        elif end != -1:
            for i in range(max(0, end - 3), end):
                candidates.append(lines[i])

        # Filter and score candidates
        best_candidate = None
        max_score = 0

        for c in candidates:
            t = c["text"].strip()
            
            # Strip common labels
            t = re.sub(r'^(NAME\s*[:\-])?\s*', '', t, flags=re.I).strip()
            
            u = t.upper()
            
            # Hard Constraints
            if len(t) < 3: continue
            if re.search(r'\d', t): continue
            
            # Filter standard headers/keywords
            skip_words = ["INCOME", "TAX", "INDIA", "GOVT", "GOVERNMENT", "PERMANENT", 
                         "ACCOUNT", "NUMBER", "CARD", "FATHER", "DATE", "BIRTH"]
            if any(x in u for x in skip_words):
                continue
            
            # Scorable Heuristics
            score = 10
            if " " in t: score += 5  # Names usually have spaces
            if len(t) < 4: score -= 2
            if t.islower(): score -= 5
            if t.isupper(): score += 2  # PAN names are uppercase
            
            if score > max_score:
                max_score = score
                best_candidate = {"value": t, "confidence": c["confidence"]}

        if best_candidate:
            return best_candidate

        # Strategy 3: spaCy NER fallback
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

    def _extract_gstin_comprehensive(self, lines, text: str, ocr_results: list = None) -> Optional[dict]:
        """Reconstructed GSTIN logic: Layout -> ML -> Fuzzy Regex -> Repair."""
        candidates = []
        
        # 1. OCR Layout Analysis (Best for structured docs)
        if ocr_results:
             layout_cands = self._find_gstin_layout(ocr_results)
             candidates.extend(layout_cands)
             
        # 2. ML/Entity Extraction (Spacy)
        ml_gstin = self._find_gstin_ml(text)
        if ml_gstin: candidates.append(ml_gstin)
        
        # 3. Fuzzy Regex (Catch OCR errors)
        fuzzy_gstin = self._find_gstin_fuzzy(text)
        if fuzzy_gstin: candidates.append(fuzzy_gstin)

        # 4. Legacy Strategies (Anchor/Spaced)
        anchor_gstin = self._find_gstin_by_anchor(lines)
        if anchor_gstin: candidates.append(anchor_gstin)
        
        spaced_gstin = self._find_gstin_spaced(text)
        if spaced_gstin: candidates.append(spaced_gstin)
        
        # Select best candidate
        best = None
        best_score = 0
        
        for cand in candidates:
            # Validate format strictly
            score = cand['confidence']
            val = cand['value']
            
            # Bonus for valid checksum
            if self._validate_gstin_checksum(val):
                score += 0.2
            
            if score > best_score:
                best_score = score
                best = cand
                
        return best

    def _find_gstin_layout(self, ocr) -> List[dict]:
        """Find GSTIN using spatial layout (right of or below 'GSTIN' label)."""
        candidates = []
        labels = ["GSTIN", "GSTIN/UIN", "GST NUMBER"]
        
        for i, (box, text, conf) in enumerate(ocr):
            # Check if this box is a label
            if any(l in text.upper() for l in labels):
                # Search for values to the RIGHT or BELOW
                label_box = box
                
                # Scan other boxes
                for j, (v_box, v_text, v_conf) in enumerate(ocr):
                    if i == j: continue
                    
                    # Clean text
                    clean_val = re.sub(r'[^A-Z0-9]', '', v_text.upper())
                    if len(clean_val) < 14: continue
                    
                    # Check spatial relationship
                    if self._is_right_of(label_box, v_box) or self._is_below(label_box, v_box):
                        # Try repair
                        repaired = self._repair_gstin_candidate(clean_val[:15])
                        if repaired:
                            candidates.append({"value": repaired, "confidence": v_conf})
        return candidates

    def _is_right_of(self, label_box, val_box):
        # Check Y-overlap and X-position
        ly_min = min(p[1] for p in label_box)
        ly_max = max(p[1] for p in label_box)
        lx_max = max(p[0] for p in label_box)
        
        vy_center = sum(p[1] for p in val_box) / 4
        vx_min = min(p[0] for p in val_box)
        
        # Value must be to the right of label
        if not (vx_min > lx_max): return False
        
        # Value center must be within label Y-range (approx)
        return ly_min - 10 <= vy_center <= ly_max + 10

    def _is_below(self, label_box, val_box):
        # Check X-overlap and Y-position
        lx_min = min(p[0] for p in label_box)
        lx_max = max(p[0] for p in label_box)
        ly_max = max(p[1] for p in label_box)
        
        vx_center = sum(p[0] for p in val_box) / 4
        vy_min = min(p[1] for p in val_box)
        
        # Value must be below label
        if not (vy_min > ly_max): return False
        
        # Value must be close (within 50px)
        if vy_min > ly_max + 100: return False
        
        # Value center must be roughly aligned with label X
        return lx_min - 20 <= vx_center <= lx_max + 100

    def _find_gstin_ml(self, text: str) -> Optional[dict]:
        """Use Spacy Entity Ruler."""
        if not self.nlp: return None
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ == "GSTIN":
                return {"value": ent.text, "confidence": 0.85}
        return None

    def _find_gstin_fuzzy(self, text: str) -> Optional[dict]:
        """Fuzzy regex search for GSTIN-like patterns."""
        # Simplified fuzzy scan
        candidates = re.findall(r'[A-Z0-9OIZSBL]{15}', text.upper())
        for cand in candidates:
            repaired = self._repair_gstin_candidate(cand)
            if repaired:
                return {"value": repaired, "confidence": 0.80}
        return None
    
    def _find_gstin_direct(self, text: str) -> Optional[dict]:
        """Direct pattern match on full text."""
        # Clean text for pattern matching
        clean_text = text.upper()
        
        # strict pattern
        m = re.search(r'\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]', clean_text)
        if m:
            return {"value": m.group(0), "confidence": 0.98}
        
        return None
    
    def _find_gstin_spaced(self, text: str) -> Optional[dict]:
        """Handle GSTIN with spaces inserted by OCR."""
        # Allow spaces between characters
        spaced_pattern = (
            r'(\d)\s*(\d)\s*'  # State code
            r'([A-Z])\s*([A-Z])\s*([A-Z])\s*([A-Z])\s*([A-Z])\s*'  # PAN letters
            r'(\d)\s*(\d)\s*(\d)\s*(\d)\s*'  # PAN digits
            r'([A-Z])\s*'  # PAN check
            r'([A-Z0-9])\s*'  # Entity code
            r'(Z)\s*'  # Default Z
            r'([A-Z0-9])'  # Check digit
        )
        m = re.search(spaced_pattern, text.upper())
        if m:
            gstin = ''.join(m.groups())
            if re.match(r'\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]', gstin):
                return {"value": gstin, "confidence": 0.90}
        
        return None
    
    def _find_gstin_by_anchor(self, lines):
        """Anchor-based GSTIN detection with context window."""
        gst_anchors = [
            ["GSTIN"],
            ["GST"],
            ["REGISTRATION", "CERTIFICATE"],
            ["GST", "REGISTRATION"],
            ["TAX", "REGISTRATION"],
        ]
        anchor_idx = -1
        for anchors in gst_anchors:
            idx = self._find_anchor_line_index(lines, anchors, True)
            if idx != -1:
                anchor_idx = idx
                break
        
        if anchor_idx == -1:
            return None
        
        # Wider context: ±3 lines
        context_start = max(0, anchor_idx - 3)
        context_end = min(len(lines), anchor_idx + 4)
        
        # Priority order: anchor line first, then nearby
        search_order = [anchor_idx]
        for offset in [1, -1, 2, -2, 3, -3]:
            candidate_idx = anchor_idx + offset
            if context_start <= candidate_idx < context_end and candidate_idx not in search_order:
                search_order.append(candidate_idx)
        
        for i in search_order:
            result = self._extract_gstin_from_line(lines[i])
            if result:
                return result
        
        return None
    
    def _find_gstin_word_scan(self, lines) -> Optional[dict]:
        """Scan each word in the document for potential GSTIN."""
        for line in lines:
            text = line["text"].upper()
            # Split by common delimiters
            words = re.split(r'[\s:;,|]+', text)
            
            for word in words:
                clean = re.sub(r'[^A-Z0-9]', '', word)
                
                # Skip if too short or too long
                if len(clean) < 15 or len(clean) > 20:
                    continue
                
                # Try direct match
                m = re.search(r'\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]', clean)
                if m:
                    return {"value": m.group(0), "confidence": line["confidence"]}
                
                # Try repair
                candidate = clean[:15]
                repaired = self._repair_gstin_candidate(candidate)
                if repaired:
                    return {"value": repaired, "confidence": line["confidence"] * 0.85}
        
        return None
    
    def _extract_gstin_from_line(self, line: dict) -> Optional[dict]:
        """Extract GSTIN from a single line with aggressive cleanup."""
        line_text = line["text"].upper()
        
        # Step 1: Remove ALL spaces first (critical for OCR with inserted spaces)
        no_space = re.sub(r'\s+', '', line_text)
        
        # Strip common GST labels
        for label in ["GSTIN", "REGISTRATION", "CERTIFICATE", "NUMBER", "NO", "GST", "UIN"]:
            no_space = re.sub(rf'{label}', '', no_space, flags=re.I)
        no_space = re.sub(r'[:\-\.\|,;]', '', no_space)
        
        # Extract alphanumeric only
        clean = re.sub(r'[^A-Z0-9]', '', no_space)
        
        # 1. Try strict pattern match
        m = re.search(r'\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]', clean)
        if m:
            return {"value": m.group(0), "confidence": line["confidence"]}
        
        # 2. Try repair at multiple starting positions (15-char candidates)
        if len(clean) >= 15:
            for start in range(min(5, len(clean) - 14)):
                candidate = clean[start:start+15]
                repaired = self._repair_gstin_candidate(candidate)
                if repaired:
                    return {"value": repaired, "confidence": line["confidence"] * 0.9}
        
        # 3. Try 14-char candidates (OCR may drop first digit)
        if len(clean) >= 14:
            for start in range(min(3, len(clean) - 13)):
                candidate = clean[start:start+14]
                repaired = self._repair_gstin_candidate(candidate)
                if repaired:
                    return {"value": repaired, "confidence": line["confidence"] * 0.85}
        
        # 4. Try any 15+ char substring
        for word in re.findall(r'[A-Z0-9]{14,}', clean):
            repaired = self._repair_gstin_candidate(word[:15] if len(word) >= 15 else word)
            if repaired:
                return {"value": repaired, "confidence": line["confidence"] * 0.9}
        
        return None
    
    def _repair_ifsc_candidate(self, word: str) -> Optional[str]:
        """Repair IFSC code (4 letters, 0, 6 alphanum)."""
        chars = list(word.upper())
        if len(chars) != 11: return None
        
        # 1. 5th character MUST be 0
        if chars[4] in ['O', 'D', 'Q', 'U', 'C', 'G']:
            chars[4] = '0'
        
        # 2. First 4 chars MUST be letters, Last 6 usually digits (but can be chars)
        # Fix common O->0 in last 6
        for i in range(5, 11):
            if chars[i] == 'O': chars[i] = '0'
            
        # 3. First 4 chars repair
        for i in range(4):
            if chars[i].isdigit():
                chars[i] = self._digit_to_char(chars[i])
                
        repaired = "".join(chars)
        if re.match(r'[A-Z]{4}0[A-Z0-9]{6}', repaired):
            return repaired
        return None

    def _repair_gstin_candidate(self, word: str) -> Optional[str]:
        """Repair OCR errors in GSTIN candidate with multiple strategies."""
        if len(word) != 15 and len(word) != 14:
            return None
            
        # Handle 14-char
        if len(word) == 14:
             for prefix in ['0', '1', '2', '3']:
                res = self._repair_gstin_15char(prefix + word)
                if res and self._validate_gstin_checksum(res): return res
             for suffix in ['Z', '5', '0']:
                 res = self._repair_gstin_15char(word + suffix)
                 if res and self._validate_gstin_checksum(res): return res
             return None
        repaired = self._repair_gstin_15char(word)
        if repaired and self._validate_gstin_checksum(repaired):
            return repaired
        variants = self._generate_gstin_variants(repaired or word)
        for cand in variants:
            if self._validate_gstin_checksum(cand) and re.match(r'\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]', cand):
                return cand
                
        return repaired
    
    def _repair_gstin_15char(self, word: str) -> Optional[str]:
        """Repair 15-char GSTIN candidate with multiple strategies."""
        c1 = self._try_repair_gstin(word, use_z_as_7=True)
        if c1 and self._validate_gstin_checksum(c1): return c1
        c2 = self._try_repair_gstin(word, use_z_as_7=False)
        if c2 and self._validate_gstin_checksum(c2): return c2
        c3 = self._try_repair_gstin(word, use_z_as_7=True, use_g_as_9=True)
        if c3 and self._validate_gstin_checksum(c3): return c3
        return c1 or c2
    
    def _try_repair_gstin(self, word: str, use_z_as_7: bool = False, use_g_as_9: bool = False) -> Optional[str]:
        """Single repair attempt with specified mappings."""
        chars = list(word.upper())
        
        if use_z_as_7:
            digit_map = {'O':'0', 'I':'1', 'L':'1', 'S':'5', 'B':'8', 'Z':'7', 'G':'6', 'D':'0', 'Q':'0', 'T':'7'}
        else:
            digit_map = {'O':'0', 'I':'1', 'L':'1', 'S':'5', 'B':'8', 'Z':'2', 'G':'6', 'D':'0', 'Q':'0', 'T':'7'}
        if use_g_as_9:
            digit_map['G'] = '9'
        
        char_map = {'0':'O', '1':'I', '5':'S', '8':'B', '2':'Z', '6':'G', '4':'A', '7':'T', '3':'E', '9':'P'}
        
        # 1. State Code (positions 0-1) -> Must be digits
        for i in [0, 1]:
            if chars[i].isalpha():
                chars[i] = digit_map.get(chars[i], chars[i])
        
        # 2. PAN Part (positions 2-6) -> Must be letters
        for i in range(2, 7):
            if chars[i].isdigit():
                chars[i] = char_map.get(chars[i], chars[i])
        
        # 3. PAN Part (positions 7-10) -> Must be digits
        for i in range(7, 11):
            if chars[i].isalpha():
                # Force specific repairs for PAN digits
                if chars[i] == 'O': chars[i] = '0'
                elif chars[i] == 'S': chars[i] = '5'
                elif chars[i] == 'I': chars[i] = '1'
                elif chars[i] == 'Z': chars[i] = '2'
                elif chars[i] == 'B': chars[i] = '8'
                else:
                    chars[i] = digit_map.get(chars[i], chars[i])
        
        # 4. PAN Check character (position 11) -> Must be letter
        if chars[11].isdigit():
            chars[11] = char_map.get(chars[11], chars[11])
        
        # 5. Entity code (position 12) -> Alphanumeric, no change
        
        # 6. Position 13 must be 'Z'
        if chars[13] != 'Z':
            if chars[13] in ['2', 'z', 'S', '5']:
                chars[13] = 'Z'
            else:
                return None
        
        # 7. Check digit (position 14) -> Alphanumeric, no change
        
        repaired = "".join(chars)
        
        # Final validation - format check only
        if re.match(r'\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]', repaired):
            return repaired
        return None
    
    def _generate_gstin_variants(self, word: str) -> List[str]:
        """Generate variants by swapping ambiguous characters (Q/O, Z/7, etc) then validation."""
        if not word or len(word) != 15: return []
        
        # Positions to try swapping:
        # 7-10 (Digits): O<->0, I<->1, Z<->2, S<->5, B<->8
        # 11 (PAN Check): Q<->0, O<->0
        # 12 (Entity): Z<->7, Z<->2
        # 14 (Check): O<->0

        candidates = set([word])
        chars = list(word)
        
        # 1. Fix PAN Digits (7-10) - Force valid digits if they are letters
        # For Z, it could be 2 or 7. Z looks more like 7 in standard fonts.
        base_chars = list(chars)
        z_indices = []
        for i in range(7, 11):
             if base_chars[i] == 'O': base_chars[i] = '0'
             if base_chars[i] == 'I': base_chars[i] = '1'
             if base_chars[i] == 'S': base_chars[i] = '5'
             if base_chars[i] == 'B': base_chars[i] = '8'
             if base_chars[i] == 'A': base_chars[i] = '4'
             if base_chars[i] == 'Z': 
                 base_chars[i] = '7' # Default to 7 (looks more like Z than 2)
                 z_indices.append(i)
        
        # Force Check Digit (14) O -> 0 as default (more common error)
        if len(base_chars) > 14 and base_chars[14] == 'O':
            base_chars[14] = '0'
            
        candidates.add("".join(base_chars))

        # 2. Try single-char swaps for likely errors
        variations = []
        
        # State Code (0-1): O/Q/D -> 0, I/1 -> 9
        for i in range(2):
            variations.append((i, {'O':'0', 'Q':'0', 'D':'0', '0':'O', 'I':'9', '1':'9'}))

        # PAN Letters (2-6): 0/Q -> O, O -> Q, 0 -> Q
        for i in range(2, 7):
             variations.append((i, {'0':'O', 'Q':'O', 'O':'Q', '0':'Q', '5':'S', '8':'B'}))
             
        # PAN Digits (7-10): Handled by base_chars mostly
        for i in range(7, 11):
            if base_chars[i] == '7': variations.append((i, {'7':'2'})) # Try 7->2
            if base_chars[i] == '5': variations.append((i, {'5':'S', '5':'3'})) # Backoff S->3
            if base_chars[i] == '1': variations.append((i, {'1':'I', '1':'9'})) # Backoff I->9

        # PAN Check (11): O/0 -> Q, Q -> O
        variations.append((11, {'O':'Q', '0':'Q', 'Q':'O', 'D':'0'})) 
        
        # Entity (12): Z->7, 7->Z, Z->2, S->5
        variations.append((12, {'Z':'7', '7':'Z', '2':'Z', 'Z':'2', 'S':'5', '5':'S'}))
        
        # Check digit (14): O->0, 0->O
        variations.append((14, {'0':'O', 'O':'0', 'Q':'0'})) 
        
        # Add Z->2 swaps for detected Zs in PAN digits (reverting base change)
        for z_idx in z_indices:
             variations.append((z_idx, {'7':'2'}))
        
        current_pool = list(candidates)
        final_candidates = []
        
        # Apply edits
        for cand in current_pool:
            c_list = list(cand)
            # Add base
            final_candidates.append(cand)
            
            # Try single swaps
            for idx, map_dict in variations:
                orig = c_list[idx]
                if orig in map_dict:
                    new_list = list(c_list)
                    new_list[idx] = map_dict[orig]
                    final_candidates.append("".join(new_list))
                    
        return final_candidates
    
    def _validate_gstin_checksum(self, gstin: str) -> bool:
        """
        Validate GSTIN checksum using official algorithm.
        
        Algorithm:
        1. Convert each char to value (0-9 = 0-9, A-Z = 10-35)
        2. For each position, multiply by factor (1 for odd, 2 for even)
        3. Divide by 36, add quotient and remainder
        4. Sum all hashes, mod 36, subtract from 36
        5. Result should match position 15
        """
        if len(gstin) != 15:
            return False
        
        char_values = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        
        try:
            total = 0
            for i, char in enumerate(gstin[:14]):
                value = char_values.index(char.upper())
                factor = 1 if i % 2 == 0 else 2
                product = value * factor
                quotient = product // 36
                remainder = product % 36
                total += quotient + remainder
            
            remainder = total % 36
            checksum_value = (36 - remainder) % 36
            expected_char = char_values[checksum_value]
            
            return gstin[14].upper() == expected_char
        except (ValueError, IndexError):
            return False
    
    def _try_fix_checksum(self, gstin: str) -> Optional[str]:
        """Try common OCR corrections at position 14 to fix checksum."""
        char_values = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        
        # Calculate expected checksum
        try:
            total = 0
            for i, char in enumerate(gstin[:14]):
                value = char_values.index(char.upper())
                factor = 1 if i % 2 == 0 else 2
                product = value * factor
                total += (product // 36) + (product % 36)
            
            checksum_value = (36 - (total % 36)) % 36
            correct_char = char_values[checksum_value]
            
            # Replace position 14 with correct checksum
            fixed = gstin[:14] + correct_char
            if re.match(r'\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]', fixed):
                return fixed
        except (ValueError, IndexError):
            pass
        
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
                    if difflib.get_close_matches(kw, words, cutoff=0.75):
                        hits += 1
            if hits == len(keywords):
                return i
        return -1

    def _extract_gst_field(self, lines, labels, stop, allowed_values: Optional[List[str]] = None):
        idx = self._find_anchor_line_index(lines, labels, True)
        if idx != -1:
            val = None
            conf = 0.0
            
            # Check same line first
            txt = lines[idx]["text"]
            
            if ":" in txt:
                cand = txt.split(":", 1)[1].strip()
                if len(cand) > 2 and not self._fuzzy_contains(cand.upper(), stop):
                    val = cand
                    conf = lines[idx]["confidence"]
            # Check for value on SAME line without delimiter (e.g. "Legal Name PARSA")
            elif " " in txt:
                 # Split by label words
                 # Remove the label part fuzzily
                 clean = txt
                 for lab in labels:
                     clean = re.sub(rf'\b{lab}\b', '', clean, flags=re.I)
                 
                 clean = clean.strip()
                 # Remove common separators
                 clean = re.sub(r'^[:\-\.]+', '', clean).strip()
                 
                 if len(clean) > 3 and not self._fuzzy_contains(clean.upper(), stop):
                     val = clean
                     conf = lines[idx]["confidence"] * 0.8
            
            # Check next line
            if not val and idx + 1 < len(lines):
                nxt = lines[idx + 1]["text"]
                if not self._fuzzy_contains(nxt.upper(), stop) and not self._fuzzy_contains(nxt.upper(), labels):
                    val = nxt
                    conf = lines[idx + 1]["confidence"]
            
            if val:
                # Fuzzy match against allowed values
                if allowed_values:
                    matches = difflib.get_close_matches(val, allowed_values, n=1, cutoff=0.6)
                    if matches:
                        return {"value": matches[0], "confidence": conf}
                
                return {"value": val, "confidence": conf}
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
