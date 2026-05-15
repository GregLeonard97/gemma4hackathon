import SwiftUI

struct NeoGuideRootView: View {
    @State private var appState: AppState
    @State private var didBootstrap = false
    @State private var isDownloadingModel = false
    @State private var downloadStartedAt: Date?
    @State private var lastProgressChangeAt: Date?
    @State private var latestDownloadProgress: Double = 0
    @State private var clockNow = Date()

    init(appState: AppState) {
        _appState = State(initialValue: appState)
    }

    var body: some View {
        ZStack {
            Color.ngBackground.ignoresSafeArea()

            switch appState.launchState {
            case .ready(let pipeline):
                NeoGuideFlowView(pipeline: pipeline)

            case .initialising:
                launchStateCard(
                    title: "Initialising",
                    subtitle: "Preparing on-device components...",
                    actionTitle: nil,
                    action: nil
                )

            case .loadingModel:
                launchStateCard(
                    title: "Loading Gemma 4 E2B",
                    subtitle: "This can take 10-30 seconds on first warm start.",
                    actionTitle: nil,
                    action: nil
                )

            case .needsModelDownload(let progress):
                launchStateCard(
                    title: "Model Download Required",
                    subtitle: modelDownloadSubtitle,
                    actionTitle: isDownloadingModel ? "Downloading..." : "Download model",
                    action: {
                        startModelDownload()
                    },
                    progress: progress,
                    progressDetail: modelDownloadProgressDetail(progress: progress),
                    showsActivityIndicator: isDownloadingModel
                )

            case .failed(let message):
                launchStateCard(
                    title: "Startup failed",
                    subtitle: message,
                    actionTitle: "Retry",
                    action: {
                        Task {
                            await appState.bootstrap()
                        }
                    }
                )
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .task {
            guard !didBootstrap else { return }
            didBootstrap = true
            await appState.bootstrap()
        }
        .onChange(of: currentDownloadProgress) { _, newValue in
            guard isDownloadingModel, let newValue else { return }

            if newValue > latestDownloadProgress + 0.0001 || newValue < latestDownloadProgress {
                latestDownloadProgress = newValue
                lastProgressChangeAt = Date()
            }
        }
        .task(id: isDownloadingModel) {
            guard isDownloadingModel else { return }

            while isDownloadingModel {
                clockNow = Date()
                try? await Task.sleep(nanoseconds: 1_000_000_000)
            }
        }
    }

    @ViewBuilder
    private func launchStateCard(
        title: String,
        subtitle: String,
        actionTitle: String?,
        action: (() -> Void)?,
        progress: Double? = nil,
        progressDetail: String? = nil,
        showsActivityIndicator: Bool = false
    ) -> some View {
        VStack(spacing: 0) {
            Spacer(minLength: 0)

            NGCard {
                VStack(alignment: .leading, spacing: 14) {
                    Text(title)
                        .font(.system(size: 18, weight: .semibold))
                        .foregroundStyle(Color.ngText)

                    Text(subtitle)
                        .font(.system(size: 13, weight: .regular))
                        .foregroundStyle(Color.ngMuted)
                        .lineSpacing(5)

                    if let progress {
                        VStack(alignment: .leading, spacing: 6) {
                            ProgressView(value: progress)
                                .tint(Color.ngAccent)
                            Text("\(Int(progress * 100))%")
                                .font(.system(size: 11, weight: .regular))
                                .foregroundStyle(Color.ngMuted)

                            if showsActivityIndicator || progressDetail != nil {
                                HStack(spacing: 8) {
                                    if showsActivityIndicator {
                                        ProgressView()
                                            .controlSize(.small)
                                    }

                                    if let progressDetail {
                                        Text(progressDetail)
                                            .font(.system(size: 11, weight: .regular))
                                            .foregroundStyle(Color.ngMuted)
                                    }
                                }
                            }
                        }
                    }

                    if let actionTitle, let action {
                        Button(action: action) {
                            Text(actionTitle)
                                .font(.system(size: 14, weight: .medium))
                                .foregroundStyle(Color.ngBackground)
                                .padding(.horizontal, 16)
                                .padding(.vertical, 10)
                                .background(Color.ngAccent, in: Capsule())
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding(20)
            }
            .padding(.horizontal, 24)
            .padding(.vertical, 24)

            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background {
            ParticleBackground()
        }
    }

    private var currentDownloadProgress: Double? {
        if case .needsModelDownload(let progress) = appState.launchState {
            return progress
        }
        return nil
    }

    private var modelDownloadSubtitle: String {
        if !isDownloadingModel {
            return "Gemma 4 E2B (~3.6 GB) must be downloaded once before first use."
        }

        if let lastProgressChangeAt,
            clockNow.timeIntervalSince(lastProgressChangeAt) > 90
        {
            return "Downloading Gemma 4 E2B (~3.6 GB). Large weight files can keep the percentage unchanged for several minutes."
        }

        return "Downloading Gemma 4 E2B (~3.6 GB). Keep the app open and on Wi-Fi."
    }

    private func modelDownloadProgressDetail(progress: Double) -> String? {
        guard isDownloadingModel, let downloadStartedAt else {
            return nil
        }

        let elapsed = formattedDuration(clockNow.timeIntervalSince(downloadStartedAt))

        if let lastProgressChangeAt {
            let stalledFor = clockNow.timeIntervalSince(lastProgressChangeAt)
            if stalledFor > 90 {
                return "Still downloading at \(Int(progress * 100))% • \(elapsed) elapsed • no change for \(formattedDuration(stalledFor))"
            }
        }

        return "\(elapsed) elapsed"
    }

    private func startModelDownload() {
        guard !isDownloadingModel else { return }

        let now = Date()
        isDownloadingModel = true
        downloadStartedAt = now
        lastProgressChangeAt = now
        latestDownloadProgress = 0
        clockNow = now

        Task {
            await appState.downloadModel { _ in }
            isDownloadingModel = false
            downloadStartedAt = nil
            lastProgressChangeAt = nil
            latestDownloadProgress = 0
        }
    }

    private func formattedDuration(_ seconds: TimeInterval) -> String {
        let clampedSeconds = max(0, Int(seconds))
        let hours = clampedSeconds / 3600
        let minutes = (clampedSeconds % 3600) / 60
        let remainingSeconds = clampedSeconds % 60

        if hours > 0 {
            return String(format: "%d:%02d:%02d", hours, minutes, remainingSeconds)
        }

        return String(format: "%d:%02d", minutes, remainingSeconds)
    }
}

private struct NeoGuideFlowView: View {
    let pipeline: RAGPipeline

    @State private var path: [Route] = []
    @State private var showSplash = true

    private enum Route: Hashable {
        case query(String)
        case pdf(Source)
    }

    var body: some View {
        NavigationStack(path: $path) {
            HomeScreen { question in
                path.append(.query(question))
            }
            .navigationDestination(for: Route.self) { route in
                switch route {
                case .query(let question):
                    QueryScreen(
                        question: question,
                        pipeline: pipeline,
                        onOpenPDF: { source in
                            path.append(.pdf(source))
                        }
                    )
                case .pdf(let source):
                    PDFViewerScreen(source: source)
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .toolbar(.hidden, for: .navigationBar)
        .overlay {
            if showSplash {
                SplashScreen {
                    showSplash = false
                }
            }
        }
    }
}
