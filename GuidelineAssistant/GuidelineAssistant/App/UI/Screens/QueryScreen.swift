import SwiftUI

struct QueryScreen: View {
    let question: String
    let pipeline: RAGPipeline
    let onOpenPDF: (Source) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var viewModel: QueryViewModel
    @State private var selectedTableImageFilename: String?

    init(question: String, pipeline: RAGPipeline, onOpenPDF: @escaping (Source) -> Void) {
        self.question = question
        self.pipeline = pipeline
        self.onOpenPDF = onOpenPDF
        _viewModel = State(initialValue: QueryViewModel(question: question))
    }

    var body: some View {
        VStack(spacing: 0) {
            NGNavigationBar(
                title: "NeoGuide",
                subtitle: "UCLH Neonatology",
                backAction: { dismiss() },
                rightLabel: sourceCountLabel
            )

            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    NGCard {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("YOUR QUESTION")
                                .font(.system(size: 11, weight: .semibold))
                                .tracking(1)
                                .foregroundStyle(Color.ngAccent)

                            Text(question)
                                .font(.system(size: 15, weight: .light))
                                .foregroundStyle(Color.ngText)
                                .lineSpacing(6)
                        }
                        .padding(.horizontal, 16)
                        .padding(.vertical, 12)
                    }

                    if case .loading = viewModel.phase, viewModel.answerText.isEmpty {
                        NGCard {
                            VStack(spacing: 14) {
                                LoadingDotsView()
                                Text("Searching UCLH guidelines...")
                                    .font(.system(size: 13, weight: .regular))
                                    .foregroundStyle(Color.ngMuted)
                            }
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 28)
                        }
                        .transition(.opacity)
                    } else {
                        answerSection
                            .transition(.opacity)
                    }

                    if !viewModel.sources.isEmpty {
                        Text("SOURCES")
                            .font(.system(size: 10, weight: .semibold))
                            .tracking(1)
                            .foregroundStyle(Color.ngMuted)
                            .padding(.top, 4)
                            .padding(.horizontal, 4)

                        VStack(spacing: 8) {
                            ForEach(viewModel.sources) { source in
                                CitationCardView(
                                    source: source,
                                    isExpanded: viewModel.expandedSourceID == source.id,
                                    onToggle: { toggleSource(source.id) },
                                    onOpenPDF: { onOpenPDF(source) }
                                )
                            }
                        }
                    }
                }
                .padding(.horizontal, 14)
                .padding(.top, 14)
                .padding(.bottom, 20)
                .animation(.easeInOut(duration: 0.3), value: viewModel.phase)
                .animation(.easeInOut(duration: 0.2), value: viewModel.expandedSourceID)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .background(
            ZStack {
                Color.ngBackground.ignoresSafeArea()
                ParticleBackground()
            }
        )
        .toolbar(.hidden, for: .navigationBar)
        .onAppear {
            viewModel.start(pipeline: pipeline)
        }
        .onDisappear {
            viewModel.stop()
        }
        .fullScreenCover(isPresented: isShowingTableSheet) {
            if let filename = selectedTableImageFilename {
                TableImageViewerScreen(imageFilename: filename)
            }
        }
    }

    private var answerSection: some View {
        Group {
            switch viewModel.phase {
            case .failed(let message):
                NGCard {
                    VStack(alignment: .leading, spacing: 10) {
                        Text("Unable to generate response")
                            .font(.system(size: 14, weight: .semibold))
                            .foregroundStyle(Color.ngText)
                        Text(message)
                            .font(.system(size: 13, weight: .regular))
                            .foregroundStyle(Color.ngMuted)
                            .lineSpacing(5)
                    }
                    .padding(16)
                }

            case .loading, .answered:
                NGCard {
                    VStack(alignment: .leading, spacing: 12) {
                        HStack(spacing: 8) {
                            RoundedRectangle(cornerRadius: 6)
                                .fill(Color.ngAccent)
                                .frame(width: 20, height: 20)
                                .overlay {
                                    Image(systemName: "sparkles")
                                        .font(.system(size: 11, weight: .semibold))
                                        .foregroundStyle(Color.ngBackground)
                                }

                            Text("AI Answer")
                                .font(.system(size: 12, weight: .medium))
                                .foregroundStyle(Color.ngAccent)
                        }

                        if viewModel.answerBlocks.isEmpty {
                            Text("Generating answer...")
                                .font(.system(size: 15, weight: .light))
                                .foregroundStyle(Color.ngMuted)
                        } else {
                            AnswerBlocksView(blocks: viewModel.answerBlocks)
                        }

                        if !viewModel.tableImageFilenames.isEmpty {
                            VStack(alignment: .leading, spacing: 8) {
                                ForEach(Array(viewModel.tableImageFilenames.enumerated()), id: \.element) { index, filename in
                                    Button {
                                        selectedTableImageFilename = filename
                                    } label: {
                                        HStack(spacing: 8) {
                                            Image(systemName: "tablecells")
                                                .font(.system(size: 12, weight: .semibold))
                                            Text(viewModel.tableImageFilenames.count > 1 ? "View full table \(index + 1)" : "View full table")
                                                .font(.system(size: 12, weight: .semibold))
                                            Spacer()
                                            Image(systemName: "arrow.up.right")
                                                .font(.system(size: 11, weight: .semibold))
                                        }
                                        .foregroundStyle(Color.ngBackground)
                                        .padding(.horizontal, 12)
                                        .padding(.vertical, 8)
                                        .background(Color.ngAccent, in: Capsule())
                                        .shadow(color: Color.ngAccent.opacity(0.12), radius: 10, x: 0, y: 0)
                                    }
                                    .buttonStyle(.plain)
                                }
                            }
                            .padding(.top, 4)
                        }
                    }
                    .padding(16)
                }
            }
        }
    }

    private var sourceCountLabel: String? {
        guard case .answered = viewModel.phase else { return nil }
        return "\(viewModel.sources.count) sources"
    }

    private func toggleSource(_ sourceID: UUID) {
        withAnimation(.easeInOut(duration: 0.2)) {
            if viewModel.expandedSourceID == sourceID {
                viewModel.expandedSourceID = nil
            } else {
                viewModel.expandedSourceID = sourceID
            }
        }
    }

    private var isShowingTableSheet: Binding<Bool> {
        Binding(
            get: { selectedTableImageFilename != nil },
            set: { isPresented in
                if !isPresented {
                    selectedTableImageFilename = nil
                }
            }
        )
    }
}
