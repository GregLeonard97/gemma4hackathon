import SwiftUI

enum AnswerBlock: Equatable {
    case paragraph(String)
    case list(heading: String?, items: [String])
    case note(String)
}

struct AnswerBlocksView: View {
    let blocks: [AnswerBlock]

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            ForEach(Array(blocks.enumerated()), id: \.offset) { _, block in
                switch block {
                case .paragraph(let text):
                    Text(markdownText(text))
                        .font(.system(size: 15, weight: .light))
                        .foregroundStyle(Color(red: 200 / 255, green: 200 / 255, blue: 208 / 255))
                        .lineSpacing(6)
                case .list(let heading, let items):
                    VStack(alignment: .leading, spacing: 6) {
                        if let heading {
                            Text(heading.uppercased())
                                .font(.system(size: 10, weight: .semibold))
                                .tracking(1)
                                .foregroundStyle(Color.ngAccent)
                        }

                        ForEach(Array(items.enumerated()), id: \.offset) { _, item in
                            HStack(alignment: .top, spacing: 8) {
                                Circle()
                                    .fill(Color.ngAccent)
                                    .frame(width: 4, height: 4)
                                    .padding(.top, 8)
                                Text(item)
                                    .font(.system(size: 15, weight: .light))
                                    .foregroundStyle(Color(red: 200 / 255, green: 200 / 255, blue: 208 / 255))
                                    .lineSpacing(6)
                            }
                        }
                    }
                case .note(let note):
                    HStack(alignment: .top, spacing: 8) {
                        Rectangle()
                            .fill(Color.ngAccent)
                            .frame(width: 2)

                        VStack(alignment: .leading, spacing: 2) {
                            Text("Tip:")
                                .font(.system(size: 12, weight: .semibold))
                                .foregroundStyle(Color.ngAccent)
                            Text(note)
                                .font(.system(size: 12, weight: .regular))
                                .foregroundStyle(Color.ngMuted)
                                .lineSpacing(4)
                        }
                        .padding(.vertical, 8)
                    }
                    .padding(.horizontal, 10)
                    .background(Color.ngNoteBg)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                }
            }
        }
    }

    private func markdownText(_ text: String) -> AttributedString {
        if let attributed = try? AttributedString(
            markdown: text,
            options: AttributedString.MarkdownParsingOptions(interpretedSyntax: .inlineOnlyPreservingWhitespace)
        ) {
            return attributed
        }
        return AttributedString(text)
    }
}
