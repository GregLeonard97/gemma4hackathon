enum TableQueryDetector {
    private static let keywords = [
        "table", "list", "schedule", "chart", "all doses", "all values",
        "complete", "full", "entire", "every row", "show me the",
        "give me the", "what are all"
    ]

    static func shouldShowTable(_ query: String) -> Bool {
        let lower = query.lowercased()
        return keywords.contains { lower.contains($0) }
    }
}
