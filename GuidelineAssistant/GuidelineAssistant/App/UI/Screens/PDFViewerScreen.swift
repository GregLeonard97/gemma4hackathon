import SwiftUI
import PDFKit

struct PDFViewerScreen: View {
    let source: Source

    @Environment(\.dismiss) private var dismiss
    private let repository = GuidelineRepository()

    var body: some View {
        VStack(spacing: 0) {
            NGNavigationBar(
                title: displayTitle,
                subtitle: "Page \(source.page)",
                backAction: { dismiss() }
            )

            HStack {
                Text("\(source.guideline) - p.\(source.page)")
                    .font(.system(size: 11, weight: .regular))
                    .foregroundStyle(Color.ngMuted)
                Spacer()
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 6)
            .background(Color.ngBackground.opacity(0.7))
            .overlay(alignment: .bottom) {
                Rectangle()
                    .fill(Color.ngBorder)
                    .frame(height: 1)
            }

            VStack(spacing: 10) {
                if let pdfURL = resolvedPDFURL {
                    NGCard {
                        PDFDocumentView(url: pdfURL, initialPage: source.page)
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .layoutPriority(1)
                } else {
                    NGCard {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("PDF unavailable")
                                .font(.system(size: 14, weight: .semibold))
                                .foregroundStyle(Color.ngText)

                            Text("Could not find \(source.guideline) in bundled guideline resources.")
                                .font(.system(size: 12, weight: .regular))
                                .foregroundStyle(Color.ngMuted)
                                .lineSpacing(5)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(14)
                    }
                }

                NGCard {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("CITED")
                            .font(.system(size: 9, weight: .semibold))
                            .tracking(0.8)
                            .foregroundStyle(Color.ngBackground)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 2)
                            .background(Color.ngAccent, in: Capsule())

                        ScrollView {
                            Text(citationExcerpt)
                                .font(.system(size: 12, weight: .regular))
                                .italic()
                                .foregroundStyle(Color.ngText)
                                .lineSpacing(6)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .frame(maxWidth: .infinity)
                        .frame(minHeight: 72, maxHeight: 180, alignment: .top)
                    }
                    .padding(12)
                    .background(Color.ngHighlight)
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(Color.ngBorderAccent, lineWidth: 1.5)
                    )
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .padding(10)
                }
                .frame(maxWidth: .infinity)
                .frame(minHeight: 150, maxHeight: 260)
                .layoutPriority(2)
            }
            .padding(14)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .background(
            ZStack {
                Color.ngBackground.ignoresSafeArea()
                ParticleBackground()
            }
        )
        .toolbar(.hidden, for: .navigationBar)
    }

    private var displayTitle: String {
        if let dot = source.guideline.lastIndex(of: ".") {
            return String(source.guideline[..<dot])
        }
        return source.guideline
    }

    private var citationExcerpt: String {
        let trimmed = source.excerpt.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            return "No excerpt available for this source reference."
        }
        return "\"\(trimmed)\""
    }

    private var resolvedPDFURL: URL? {
        if let directURL = repository.guidelinePDFURL(filename: source.guideline),
            FileManager.default.fileExists(atPath: directURL.path)
        {
            return directURL
        }

        let underscoredName = source.guideline.replacingOccurrences(of: " ", with: "_")
        if let underscoredURL = repository.guidelinePDFURL(filename: underscoredName),
            FileManager.default.fileExists(atPath: underscoredURL.path)
        {
            return underscoredURL
        }

        let normalizedName = source.guideline.replacingOccurrences(of: "_", with: " ")
        if let normalizedURL = repository.guidelinePDFURL(filename: normalizedName),
            FileManager.default.fileExists(atPath: normalizedURL.path)
        {
            return normalizedURL
        }

        return nil
    }
}

private struct PDFDocumentView: UIViewRepresentable {
    let url: URL
    let initialPage: Int

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func makeUIView(context: Context) -> PDFView {
        let pdfView = PDFView()
        configure(pdfView)
        loadDocument(into: pdfView, coordinator: context.coordinator)
        return pdfView
    }

    func updateUIView(_ uiView: PDFView, context: Context) {
        let urlChanged = context.coordinator.lastURL?.path != url.path
        let pageChanged = context.coordinator.lastRequestedPage != initialPage

        if urlChanged || pageChanged || uiView.document == nil {
            loadDocument(into: uiView, coordinator: context.coordinator)
        }
    }

    private func configure(_ pdfView: PDFView) {
        pdfView.autoScales = true
        pdfView.displayMode = .singlePageContinuous
        pdfView.displayDirection = .vertical
        pdfView.displaysPageBreaks = true
        pdfView.backgroundColor = .black
        pdfView.usePageViewController(false, withViewOptions: nil)
    }

    private func loadDocument(into pdfView: PDFView, coordinator: Coordinator) {
        guard let document = PDFDocument(url: url) else {
            coordinator.lastURL = url
            coordinator.lastRequestedPage = initialPage
            pdfView.document = nil
            return
        }

        coordinator.lastURL = url
        coordinator.lastRequestedPage = initialPage
        pdfView.document = document

        let maxPageIndex = max(0, document.pageCount - 1)
        let pageIndex = max(0, min(initialPage - 1, maxPageIndex))
        if let page = document.page(at: pageIndex) {
            pdfView.go(to: page)

            // PDFView may reset to page 1 right after document assignment; re-apply once layout settles.
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.12) { [weak pdfView, weak document] in
                guard let pdfView, let document,
                    let activeDocument = pdfView.document, activeDocument === document,
                    let delayedPage = document.page(at: pageIndex)
                else {
                    return
                }

                pdfView.go(to: delayedPage)
            }
        }
    }

    final class Coordinator {
        var lastURL: URL?
        var lastRequestedPage: Int?
    }
}
