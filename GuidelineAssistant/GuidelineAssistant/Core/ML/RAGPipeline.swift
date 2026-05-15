import Foundation
import Observation
import os

@Observable
final class RAGPipeline: @unchecked Sendable {
     private static let systemPrompt = """
You are a clinical guideline assistant for hospital staff.

CLINICAL INTERPRETATION:
When interpreting questions, use standard clinical conventions: 
"follow-up" typically refers to post-discharge surveillance and 
appointments, "monitoring" refers to inpatient observation during 
the current admission, and "criteria" usually refers to specific 
thresholds or conditions for a clinical decision.

ANSWERING RULES:
1. Answer using the provided guideline excerpts below.

2. Only refuse with "This isn't covered in the available guidelines. 
   Please consult a senior colleague." if the relevant information is 
   genuinely absent — not just because the wording differs.

3. For specific drug doses, dosing intervals, numeric thresholds, or 
   exact values: only state values that appear verbatim in the provided 
   excerpts. If the specific value the user asks for is not present, 
   refuse rather than estimate, calculate from general knowledge, or 
   extrapolate from related values.

4. Never fabricate drug names, indications, contraindications, or 
   procedures that aren't in the provided excerpts.

5. For calculations:
   (a) only use input values stated in the question or in the excerpts;
   (b) show every step of your working with units at each step;
   (c) if a required input value is missing, state what's missing and 
   refuse to calculate.

6. For patient scenarios involving age: identify whether each value 
   refers to gestational age, postnatal age, corrected age, or 
   treatment duration before answering, and state your interpretation 
   briefly at the start of your response.
   
   Default conventions when the question doesn't specify:
   - "X weeks" or "X-week" alone almost always means gestational age 
     at birth (e.g., "28 weeks" = born at 28 weeks gestation). This 
     is true for any value typically within human gestation range 
     (roughly 20-42 weeks).
   - "X days old" or "day X" means postnatal age (days since birth).
   - "X months old" means postnatal age unless explicitly stated as 
     corrected age.
   - When both gestational and postnatal age are given (e.g., "28 week, 
     day 1"), apply both correctly: gestational age determines patient 
     population (preterm/term), postnatal age determines current 
     management timing.
   
   If after applying these conventions the question still permits two 
   interpretations of age that would lead to different answers (e.g., 
   different drug doses, different management), refuse with: "I'm 
   unsure whether you mean [interpretation A] or [interpretation B], 
   and this affects [the specific clinical decision]. Please rephrase 
   with explicit ages — for example, 'gestational age X weeks, postnatal 
   age Y days'."
   
   For non-dosing questions where interpretation doesn't change the 
   answer, proceed and state the assumption clearly.

7. If the retrieved excerpts contain partial information, state what you 
   can answer from them and explicitly note what additional information 
   would be needed for completeness.

FORMAT RULES:
8. Write the answer as clean prose. Do NOT include source citations in 
   your answer text — sources are displayed separately.

9. If the user asks to see a table, list, regimen, or schedule (signal 
   phrases: "show me the...", "what are all...", "give me the table") 
   AND a [TABLE IMAGE AVAILABLE] marker is present in the context, 
   write a one-sentence summary like "The full dosing table is shown 
   below from [guideline name], page X." Do NOT reproduce the table 
   contents in text.

10. If the user asks for tabular content but no [TABLE IMAGE AVAILABLE] 
    marker is present, answer only with values explicitly in the retrieved 
    excerpts — do not reconstruct missing rows or extrapolate.

11. For specific value lookups (e.g., "amikacin dose for a 30-week baby"), 
    answer directly from the table row data without referring to images.

12. Keep answers concise and clinically actionable.
"""

    private let llm: any LLMEngineProtocol
    private let embedder: any EmbeddingEngineProtocol
    private let vectorStore: any VectorStoreProtocol
    private let repository: GuidelineRepository
    private let logger = Logger(subsystem: "GuidelineAssistant", category: "RAGPipeline")

    init(
        llm: any LLMEngineProtocol,
        embedder: any EmbeddingEngineProtocol,
        vectorStore: any VectorStoreProtocol,
        repository: GuidelineRepository = GuidelineRepository()
    ) {
        self.llm = llm
        self.embedder = embedder
        self.vectorStore = vectorStore
        self.repository = repository
    }

    func query(_ question: String) -> AsyncThrowingStream<QueryEvent, Error> {
        AsyncThrowingStream { continuation in
            Task { [llm, embedder, vectorStore, repository, logger] in
                do {
                    continuation.yield(.retrievalStarted)

                    let expanded = AbbreviationExpander.expand(question)
                    let queryEmbedding = try await embedder.embed(expanded)
                    let chunks = try vectorStore.search(queryEmbedding: queryEmbedding, topK: 5)

                    let tableImage: String?
                    if TableQueryDetector.shouldShowTable(question) {
                        tableImage = chunks.first(where: { $0.tableImage != nil })?.tableImage
                    } else {
                        tableImage = nil
                    }

                    let sources = repository.sources(from: chunks, includeTableImage: tableImage != nil)
                    continuation.yield(.retrievalComplete(sources: sources, tableImage: tableImage))

                    continuation.yield(.generationStarted)

                    let context = buildContext(chunks: chunks, tableImage: tableImage)
                    let userPrompt = """
GUIDELINE EXCERPTS:

\(context)

---

QUESTION: \(expanded)
"""

                    let stream = llm.generate(systemPrompt: Self.systemPrompt, userPrompt: userPrompt)
                    for try await token in stream {
                        continuation.yield(.token(token))
                    }

                    continuation.yield(.complete)
                    continuation.finish()
                } catch {
                    logger.error("Pipeline failed: \(error.localizedDescription)")
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    private func buildContext(chunks: [RetrievedChunk], tableImage: String?) -> String {
        var blocks: [String] = []

        if let tableImage,
           let sourceChunk = chunks.first(where: { $0.tableImage == tableImage }) {
            blocks.append(
                """
[TABLE IMAGE AVAILABLE - \(sourceChunk.source), page \(sourceChunk.page)]
A rendered table image will be displayed by the app.
"""
            )
        }

        for chunk in chunks {
            let kind: String
            switch chunk.chunkType {
            case .text:
                kind = "TEXT"
            case .tableRow, .tableSummary:
                kind = "TABLE ROW"
            }

            let sectionPart = chunk.section.map { " - \($0)" } ?? ""
            let header = "[\(kind) - \(chunk.source), Page \(chunk.page)\(sectionPart)]"
            blocks.append("\(header)\n\(chunk.content)")
        }

        return blocks.joined(separator: "\n\n---\n\n")
    }
}
