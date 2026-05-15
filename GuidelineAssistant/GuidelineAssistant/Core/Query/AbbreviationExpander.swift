import Foundation

enum AbbreviationExpander {
    private static let medicalAbbreviations: [(String, String)] = [
        ("LP", "lumbar puncture"),
        ("FBC", "full blood count"),
        ("U&E", "urea and electrolytes"),
        ("UE", "urea and electrolytes"),
        ("LFT", "liver function tests"),
        ("LFTs", "liver function tests"),
        ("CRP", "c-reactive protein"),
        ("ESR", "erythrocyte sedimentation rate"),
        ("ECG", "electrocardiogram"),
        ("EEG", "electroencephalogram"),
        ("ABG", "arterial blood gas"),
        ("VBG", "venous blood gas"),
        ("CBG", "capillary blood gas"),
        ("USS", "ultrasound scan"),
        ("CT", "computed tomography"),
        ("MRI", "magnetic resonance imaging"),
        ("CXR", "chest x-ray"),
        ("AXR", "abdominal x-ray"),
        ("MSU", "midstream urine"),
        ("MCS", "microscopy culture sensitivity"),
        ("BC", "blood culture"),
        ("MC&S", "microscopy culture and sensitivity"),
        ("KP", "Kaiser Permanente"),

        ("DKA", "diabetic ketoacidosis"),
        ("UTI", "urinary tract infection"),
        ("URTI", "upper respiratory tract infection"),
        ("LRTI", "lower respiratory tract infection"),
        ("CAP", "community acquired pneumonia"),
        ("HAP", "hospital acquired pneumonia"),
        ("RSV", "respiratory syncytial virus"),
        ("EOS", "early onset sepsis"),
        ("LOS", "late onset sepsis"),
        ("NEC", "necrotising enterocolitis"),
        ("RDS", "respiratory distress syndrome"),
        ("BPD", "bronchopulmonary dysplasia"),
        ("CLD", "chronic lung disease"),
        ("PDA", "patent ductus arteriosus"),
        ("IVH", "intraventricular haemorrhage"),
        ("HIE", "hypoxic ischaemic encephalopathy"),
        ("GBS", "group b streptococcus"),
        ("MRSA", "methicillin resistant staphylococcus aureus"),
        ("ASD", "atrial septal defect"),
        ("VSD", "ventricular septal defect"),
        ("TOF", "tetralogy of fallot"),
        ("CHD", "congenital heart disease"),
        ("TGA", "transposition of the great arteries"),
        ("CDH", "congenital diaphragmatic hernia"),
        ("GORD", "gastro-oesophageal reflux disease"),
        ("GERD", "gastro-oesophageal reflux disease"),
        ("IBD", "inflammatory bowel disease"),
        ("IBS", "irritable bowel syndrome"),
        ("JIA", "juvenile idiopathic arthritis"),
        ("T1DM", "type 1 diabetes mellitus"),
        ("T2DM", "type 2 diabetes mellitus"),
        ("NEC", "necrotising enterocolitis"),

        ("TPN", "total parenteral nutrition"),
        ("PN", "parenteral nutrition"),
        ("EN", "enteral nutrition"),
        ("NG", "nasogastric"),
        ("OG", "orogastric"),
        ("PEG", "percutaneous endoscopic gastrostomy"),
        ("ETT", "endotracheal tube"),
        ("CPAP", "continuous positive airway pressure"),
        ("BIPAP", "bilevel positive airway pressure"),
        ("HFOV", "high frequency oscillatory ventilation"),
        ("ECMO", "extracorporeal membrane oxygenation"),
        ("IVIG", "intravenous immunoglobulin"),
        ("IV", "intravenous"),
        ("IM", "intramuscular"),
        ("SC", "subcutaneous"),
        ("PO", "oral"),
        ("PR", "rectal"),
        ("BD", "twice daily"),
        ("TDS", "three times daily"),
        ("QDS", "four times daily"),
        ("OD", "once daily"),
        ("PRN", "as required"),
        ("STAT", "immediately"),

        ("CNS", "central nervous system"),
        ("CSF", "cerebrospinal fluid"),
        ("GI", "gastrointestinal"),
        ("GU", "genitourinary"),
        ("MSK", "musculoskeletal"),
        ("ENT", "ear nose and throat"),
        ("CVS", "cardiovascular system"),
        ("RS", "respiratory system"),
        ("BP", "blood pressure"),
        ("HR", "heart rate"),
        ("RR", "respiratory rate"),
        ("SpO2", "oxygen saturation"),
        ("MAP", "mean arterial pressure"),
        ("ICP", "intracranial pressure"),
        ("GCS", "glasgow coma scale"),

        ("A&E", "accident and emergency"),
        ("ED", "emergency department"),
        ("ICU", "intensive care unit"),
        ("PICU", "paediatric intensive care unit"),
        ("NICU", "neonatal intensive care unit"),
        ("HDU", "high dependency unit"),
        ("OPD", "outpatient department"),
        ("OOH", "out of hours"),
        ("MDT", "multidisciplinary team"),
        ("FTT", "failure to thrive"),
        ("SOB", "shortness of breath"),
        ("LA", "local anaesthesia"),
        ("DOA", "date of admission"),
        ("DOD", "date of discharge"),
        ("DOB", "date of birth"),
        ("GA", "gestational age"),
        ("CGA", "corrected gestational age"),
        ("LMP", "last menstrual period"),
        ("EDD", "estimated date of delivery"),
        ("SpR", "registrar"),
        ("SHO", "senior house officer"),
        ("FY1", "foundation year 1"),
        ("FY2", "foundation year 2"),
        ("registrar", "SpR"),

        ("PCM", "paracetamol"),
        ("NSAID", "non-steroidal anti-inflammatory drug"),
        ("PPI", "proton pump inhibitor"),
        ("ABX", "antibiotics"),
        ("Abx", "antibiotics")
    ]

    private static let abbreviationToFull: [(String, String)] = {
        var seen = Set<String>()
        var ordered: [(String, String)] = []

        for (abbreviation, full) in medicalAbbreviations {
            let lower = abbreviation.lowercased()
            guard !seen.contains(lower) else { continue }
            seen.insert(lower)
            ordered.append((lower, full))
        }

        return ordered
    }()

    private static let fullToAbbreviation: [(String, String)] = {
        var seen = Set<String>()
        var ordered: [(String, String)] = []

        for (abbreviation, full) in medicalAbbreviations {
            let fullLower = full.lowercased()
            guard !seen.contains(fullLower) else { continue }
            seen.insert(fullLower)
            ordered.append((fullLower, abbreviation))
        }

        return ordered
    }()

    static func expand(_ query: String) -> String {
        var expanded = query

        let abbreviationPairs = abbreviationToFull.sorted { lhs, rhs in
            lhs.0.count > rhs.0.count
        }

        for (abbreviationLower, full) in abbreviationPairs {
            let pattern = "(?<![A-Za-z0-9])\(NSRegularExpression.escapedPattern(for: abbreviationLower))(?![A-Za-z0-9])"
            expanded = replacingMatches(in: expanded, pattern: pattern) { match in
                "\(match) (\(full))"
            }
        }

        let fullPairs = fullToAbbreviation.sorted { lhs, rhs in
            lhs.0.count > rhs.0.count
        }

        for (fullLower, abbreviation) in fullPairs {
            guard fullLower.contains(" ") else { continue }
            guard !expanded.lowercased().contains(abbreviation.lowercased()) else { continue }

            let pattern = NSRegularExpression.escapedPattern(for: fullLower)
            expanded = replacingFirstMatch(in: expanded, pattern: pattern) { match in
                "\(match) (\(abbreviation))"
            }
        }

        return expanded
    }

    private static func replacingMatches(
        in text: String,
        pattern: String,
        replacement: (String) -> String
    ) -> String {
        guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else {
            return text
        }

        let nsRange = NSRange(text.startIndex..<text.endIndex, in: text)
        let matches = regex.matches(in: text, options: [], range: nsRange)
        if matches.isEmpty { return text }

        var result = text
        for match in matches.reversed() {
            guard let range = Range(match.range, in: result) else { continue }
            let matched = String(result[range])
            result.replaceSubrange(range, with: replacement(matched))
        }
        return result
    }

    private static func replacingFirstMatch(
        in text: String,
        pattern: String,
        replacement: (String) -> String
    ) -> String {
        guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else {
            return text
        }

        let nsRange = NSRange(text.startIndex..<text.endIndex, in: text)
        guard let match = regex.firstMatch(in: text, options: [], range: nsRange),
              let range = Range(match.range, in: text) else {
            return text
        }

        var result = text
        let matched = String(result[range])
        result.replaceSubrange(range, with: replacement(matched))
        return result
    }
}
