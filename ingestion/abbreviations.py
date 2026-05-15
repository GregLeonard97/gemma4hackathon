"""
Medical abbreviation handling for query expansion.

Maintains a dictionary of common pediatric/clinical abbreviations and
provides a function to expand abbreviations in user queries before
embedding. This dramatically improves retrieval recall when guidelines
use full terms but users search with abbreviations (or vice versa).

Usage in test_generation.py:
    from abbreviations import expand_query
    
    expanded = expand_query(user_question)
    chunks = retrieve(collection, expanded)
"""

import re

# Comprehensive pediatric/neonatal/general medical abbreviations.
# Format: 'abbreviation': 'expansion'
# Bidirectional matching is handled at runtime.
#
# Add hospital-specific abbreviations as you encounter them.
MEDICAL_ABBREVIATIONS = {
    # Procedures and investigations
    "LP": "lumbar puncture",
    "FBC": "full blood count",
    "U&E": "urea and electrolytes",
    "UE": "urea and electrolytes",
    "LFT": "liver function tests",
    "LFTs": "liver function tests",
    "CRP": "c-reactive protein",
    "ESR": "erythrocyte sedimentation rate",
    "ECG": "electrocardiogram",
    "EEG": "electroencephalogram",
    "ABG": "arterial blood gas",
    "VBG": "venous blood gas",
    "CBG": "capillary blood gas",
    "USS": "ultrasound scan",
    "CT": "computed tomography",
    "MRI": "magnetic resonance imaging",
    "CXR": "chest x-ray",
    "AXR": "abdominal x-ray",
    "MSU": "midstream urine",
    "MCS": "microscopy culture sensitivity",
    "BC": "blood culture",
    "MC&S": "microscopy culture and sensitivity",
    "KP": "Kaiser Permanente",
    
    # Conditions
    "DKA": "diabetic ketoacidosis",
    "UTI": "urinary tract infection",
    "URTI": "upper respiratory tract infection",
    "LRTI": "lower respiratory tract infection",
    "CAP": "community acquired pneumonia",
    "HAP": "hospital acquired pneumonia",
    "RSV": "respiratory syncytial virus",
    "EOS": "early onset sepsis",
    "LOS": "late onset sepsis",
    "NEC": "necrotising enterocolitis",
    "RDS": "respiratory distress syndrome",
    "BPD": "bronchopulmonary dysplasia",
    "CLD": "chronic lung disease",
    "PDA": "patent ductus arteriosus",
    "IVH": "intraventricular haemorrhage",
    "HIE": "hypoxic ischaemic encephalopathy",
    "GBS": "group b streptococcus",
    "MRSA": "methicillin resistant staphylococcus aureus",
    "ASD": "atrial septal defect",
    "VSD": "ventricular septal defect",
    "TOF": "tetralogy of fallot",
    "CHD": "congenital heart disease",
    "TGA": "transposition of the great arteries",
    "CDH": "congenital diaphragmatic hernia",
    "GORD": "gastro-oesophageal reflux disease",
    "GERD": "gastro-oesophageal reflux disease",
    "IBD": "inflammatory bowel disease",
    "IBS": "irritable bowel syndrome",
    "JIA": "juvenile idiopathic arthritis",
    "T1DM": "type 1 diabetes mellitus",
    "T2DM": "type 2 diabetes mellitus",
    "NEC": "necrotising enterocolitis",
    
    # Treatments / management
    "TPN": "total parenteral nutrition",
    "PN": "parenteral nutrition",
    "EN": "enteral nutrition",
    "NG": "nasogastric",
    "OG": "orogastric",
    "PEG": "percutaneous endoscopic gastrostomy",
    "ETT": "endotracheal tube",
    "CPAP": "continuous positive airway pressure",
    "BIPAP": "bilevel positive airway pressure",
    "HFOV": "high frequency oscillatory ventilation",
    "ECMO": "extracorporeal membrane oxygenation",
    "IVIG": "intravenous immunoglobulin",
    "IV": "intravenous",
    "IM": "intramuscular",
    "SC": "subcutaneous",
    "PO": "oral",
    "PR": "rectal",
    "BD": "twice daily",
    "TDS": "three times daily",
    "QDS": "four times daily",
    "OD": "once daily",
    "PRN": "as required",
    "STAT": "immediately",
    
    # Anatomy / physiology
    "CNS": "central nervous system",
    "CSF": "cerebrospinal fluid",
    "GI": "gastrointestinal",
    "GU": "genitourinary",
    "MSK": "musculoskeletal",
    "ENT": "ear nose and throat",
    "CVS": "cardiovascular system",
    "RS": "respiratory system",
    "BP": "blood pressure",
    "HR": "heart rate",
    "RR": "respiratory rate",
    "SpO2": "oxygen saturation",
    "MAP": "mean arterial pressure",
    "ICP": "intracranial pressure",
    "GCS": "glasgow coma scale",
    
    # Clinical context
    "A&E": "accident and emergency",
    "ED": "emergency department",
    "ICU": "intensive care unit",
    "PICU": "paediatric intensive care unit",
    "NICU": "neonatal intensive care unit",
    "HDU": "high dependency unit",
    "OPD": "outpatient department",
    "OOH": "out of hours",
    "MDT": "multidisciplinary team",
    "FTT": "failure to thrive",
    "SOB": "shortness of breath",
    "LA": "local anaesthesia",
    "DOA": "date of admission",
    "DOD": "date of discharge",
    "DOB": "date of birth",
    "GA": "gestational age",
    "CGA": "corrected gestational age",
    "LMP": "last menstrual period",
    "EDD": "estimated date of delivery",
    "SpR": "registrar",
    "SHO": "senior house officer",
    "FY1": "foundation year 1",
    "FY2": "foundation year 2",
    "registrar": "SpR",
    
    # Pharmacology
    "PCM": "paracetamol",
    "NSAID": "non-steroidal anti-inflammatory drug",
    "PPI": "proton pump inhibitor",
    "ABX": "antibiotics",
    "Abx": "antibiotics",
}


def _build_lookup_maps():
    """Build bidirectional lookup maps, normalised to lowercase."""
    abbrev_to_full = {}
    full_to_abbrev = {}
    
    for abbrev, full in MEDICAL_ABBREVIATIONS.items():
        # Lowercase for case-insensitive matching
        abbrev_lower = abbrev.lower()
        full_lower = full.lower()
        
        # If multiple abbreviations map to the same full form, keep both
        if abbrev_lower not in abbrev_to_full:
            abbrev_to_full[abbrev_lower] = full
        
        if full_lower not in full_to_abbrev:
            full_to_abbrev[full_lower] = abbrev
    
    return abbrev_to_full, full_to_abbrev


_ABBREV_TO_FULL, _FULL_TO_ABBREV = _build_lookup_maps()


def expand_query(query: str) -> str:
    """
    Expand medical abbreviations and full terms in a query.
    
    Adds expansions alongside the original term so embedding captures both:
        "LP indications" -> "LP (lumbar puncture) indications"
        "lumbar puncture indications" -> "lumbar puncture (LP) indications"
    
    This makes retrieval robust to whether the guideline or the user
    used the abbreviation vs the full form.
    """
    expanded = query
    
    # Match whole-word abbreviations (case-insensitive but capture original case)
    # Sort by length descending to handle multi-word matches first
    
    # Expand abbreviations -> add full form
    # Use word boundaries to avoid matching inside other words
    for abbrev_lower, full in sorted(
        _ABBREV_TO_FULL.items(),
        key=lambda x: -len(x[0])
    ):
        # Build a regex that matches the abbreviation as a whole word
        # \b doesn't work for "U&E" because of the &, so we use a custom boundary
        pattern = (
            r'(?<![a-zA-Z0-9])' +  # Not preceded by alphanumeric
            re.escape(abbrev_lower) + 
            r'(?![a-zA-Z0-9])'      # Not followed by alphanumeric
        )
        
        def replace_match(match):
            original = match.group(0)
            # Don't expand if expansion already nearby in query
            return f"{original} ({full})"
        
        expanded = re.sub(
            pattern,
            replace_match,
            expanded,
            flags=re.IGNORECASE
        )
    
    # Expand full forms -> add abbreviation
    # (Only for multi-word full forms to avoid matching common words)
    for full_lower, abbrev in sorted(
        _FULL_TO_ABBREV.items(),
        key=lambda x: -len(x[0])
    ):
        if " " not in full_lower:
            continue  # Skip single-word expansions
        
        pattern = re.escape(full_lower)
        
        def replace_match(match):
            original = match.group(0)
            return f"{original} ({abbrev})"
        
        # Only expand if the abbreviation isn't already present
        if abbrev.lower() not in expanded.lower():
            expanded = re.sub(
                pattern,
                replace_match,
                expanded,
                flags=re.IGNORECASE,
                count=1  # Expand only first occurrence to avoid duplication
            )
    
    return expanded


# Quick self-test
if __name__ == "__main__":
    test_queries = [
        "indications for LP in suspected meningitis",
        "lumbar puncture procedure",
        "CRP cutoff for sepsis",
        "follow-up after DKA admission",
        "TPN protein requirements day 3",
        "what's the gentamicin dose for a 28 week neonate",
        "RSV bronchiolitis management",
    ]
    
    print("Query Expansion Tests:\n")
    for q in test_queries:
        expanded = expand_query(q)
        if expanded != q:
            print(f"  Original: {q}")
            print(f"  Expanded: {expanded}")
            print()
        else:
            print(f"  Unchanged: {q}\n")