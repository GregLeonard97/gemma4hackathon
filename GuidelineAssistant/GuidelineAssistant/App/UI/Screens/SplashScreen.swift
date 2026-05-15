import SwiftUI

struct SplashScreen: View {
    let onDone: () -> Void

    @State private var hasAppeared = false
    @State private var isLeaving = false
    @State private var hasCompleted = false

    var body: some View {
        GeometryReader { proxy in
            ZStack {
                Color.ngBackground.ignoresSafeArea()
                ParticleBackground()

                VStack(spacing: 0) {
                    VStack(spacing: 0) {
                        RoundedRectangle(cornerRadius: 20)
                            .fill(Color.ngAccent.opacity(0.08))
                            .overlay(
                                RoundedRectangle(cornerRadius: 20)
                                    .stroke(Color.ngBorderAccent, lineWidth: 1)
                            )
                            .overlay {
                                ZStack {
                                    Circle().stroke(Color.ngAccent.opacity(0.4), lineWidth: 1.4)
                                        .frame(width: 42, height: 42)
                                    Circle().stroke(Color.ngAccent.opacity(0.7), lineWidth: 1.4)
                                        .frame(width: 26, height: 26)
                                    Circle().fill(Color.ngAccent)
                                        .frame(width: 8, height: 8)
                                }
                            }
                            .frame(width: 72, height: 72)
                            .shadow(color: Color.ngAccent.opacity(0.15), radius: 25, x: 0, y: 10)

                        Text("NeoGuide")
                            .font(.system(size: 42, weight: .ultraLight))
                            .tracking(-1.2)
                            .foregroundStyle(Color.ngText)
                            .padding(.top, 28)

                        Text("UCLH NEONATOLOGY")
                            .font(.system(size: 12, weight: .regular))
                            .tracking(1.2)
                            .foregroundStyle(Color.ngAccent)
                            .padding(.top, 6)

                        Text("Clinical guideline retrieval powered by AI - built for the NICU.")
                            .font(.system(size: 14, weight: .light))
                            .foregroundStyle(Color.white.opacity(0.65))
                            .multilineTextAlignment(.center)
                            .lineSpacing(6)
                            .lineLimit(nil)
                            .fixedSize(horizontal: false, vertical: true)
                            .frame(maxWidth: 320)
                            .padding(.top, 32)
                    }

                    Spacer(minLength: 20)

                    Button(action: beginExit) {
                        HStack(spacing: 6) {
                            Text("Get started")
                                .font(.system(size: 15, weight: .medium))
                            Image(systemName: "arrow.right")
                                .font(.system(size: 12, weight: .semibold))
                        }
                        .foregroundStyle(Color.ngBackground)
                        .padding(.horizontal, 36)
                        .padding(.vertical, 13)
                        .background(Color.ngAccent, in: Capsule())
                    }

                    Text("For use by UCLH clinical staff only. Always apply clinical judgement.")
                        .font(.system(size: 10.5, weight: .regular))
                        .foregroundStyle(Color.white.opacity(0.2))
                        .multilineTextAlignment(.center)
                        .lineSpacing(4)
                        .frame(maxWidth: 280)
                        .padding(.top, 18)
                }
                .padding(.horizontal, 32)
                .padding(.top, max(28, proxy.safeAreaInsets.top + 12))
                .padding(.bottom, max(26, proxy.safeAreaInsets.bottom + 12))
                .frame(width: proxy.size.width, height: proxy.size.height, alignment: .top)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .opacity(isLeaving ? 0 : (hasAppeared ? 1 : 0))
        .animation(.easeInOut(duration: isLeaving ? 0.5 : 0.8), value: isLeaving)
        .animation(.easeInOut(duration: 0.8), value: hasAppeared)
        .task {
            if !hasAppeared {
                hasAppeared = true
            }

            try? await Task.sleep(for: .milliseconds(3500))
            beginExit()
        }
    }

    private func beginExit() {
        guard !hasCompleted else { return }
        hasCompleted = true
        isLeaving = true

        Task {
            try? await Task.sleep(for: .milliseconds(500))
            onDone()
        }
    }
}
