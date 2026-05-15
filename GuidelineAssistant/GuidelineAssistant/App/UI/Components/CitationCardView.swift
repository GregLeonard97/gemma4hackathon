import SwiftUI

struct CitationCardView: View {
    let source: Source
    let isExpanded: Bool
    let onToggle: () -> Void
    let onOpenPDF: () -> Void

    var body: some View {
        NGCard {
            VStack(spacing: 0) {
                HStack(spacing: 10) {
                    RoundedRectangle(cornerRadius: 10)
                        .fill(Color.ngSourceIcon)
                        .overlay(
                            RoundedRectangle(cornerRadius: 10)
                                .stroke(Color.ngBorderAccent, lineWidth: 1)
                        )
                        .overlay {
                            Image(systemName: "doc.text")
                                .font(.system(size: 14, weight: .medium))
                                .foregroundStyle(Color.ngAccent)
                        }
                        .frame(width: 30, height: 30)

                    VStack(alignment: .leading, spacing: 2) {
                        Text(displayTitle)
                            .font(.system(size: 13, weight: .regular))
                            .foregroundStyle(Color.ngText)
                            .lineLimit(1)
                        Text("p.\(source.page)")
                            .font(.system(size: 11, weight: .regular))
                            .foregroundStyle(Color.ngMuted)
                    }

                    Spacer()

                    Button(action: onToggle) {
                        Image(systemName: "chevron.down")
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundStyle(Color.ngMuted)
                            .rotationEffect(.degrees(isExpanded ? 180 : 0))
                            .animation(.easeInOut(duration: 0.2), value: isExpanded)
                    }
                    .buttonStyle(.plain)
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 11)
                .contentShape(Rectangle())
                .onTapGesture {
                    onOpenPDF()
                }

                if isExpanded {
                    VStack(alignment: .leading, spacing: 10) {
                        Rectangle()
                            .fill(Color.ngBorder)
                            .frame(height: 1)

                        Text("\"\(source.excerpt)\"")
                            .font(.system(size: 13, weight: .regular))
                            .italic()
                            .foregroundStyle(Color.ngMuted)
                            .lineSpacing(5)

                        Button(action: onOpenPDF) {
                            HStack(spacing: 6) {
                                Image(systemName: "doc.text")
                                    .font(.system(size: 11, weight: .semibold))
                                Text("Open PDF - p.\(source.page)")
                                    .font(.system(size: 12, weight: .semibold))
                            }
                            .foregroundStyle(Color.ngBackground)
                            .padding(.horizontal, 14)
                            .padding(.vertical, 6)
                            .background(Color.ngAccent, in: Capsule())
                            .shadow(color: Color.ngAccent.opacity(0.12), radius: 10, x: 0, y: 0)
                        }
                        .buttonStyle(.plain)
                    }
                    .padding(.horizontal, 14)
                    .padding(.top, 8)
                    .padding(.bottom, 12)
                    .transition(.opacity.combined(with: .move(edge: .top)))
                }
            }
        }
    }

    private var displayTitle: String {
        let name = source.guideline
        if let dot = name.lastIndex(of: ".") {
            return String(name[..<dot])
        }
        return name
    }
}
