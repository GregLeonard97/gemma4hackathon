import SwiftUI
import UIKit

struct HomeScreen: View {
    let onSubmit: (String) -> Void

    @State private var query = ""
    @State private var showDebugLog = false
    @FocusState private var isQueryFocused: Bool

    private let suggestions = [
        "Surfactant dosing",
        "Caffeine loading",
        "TPN composition",
        "Phototherapy threshold"
    ]

    var body: some View {
        VStack(spacing: 0) {
            NGNavigationBar(
                title: "NeoGuide",
                subtitle: "UCLH Neonatology",
                rightSystemImage: "ladybug",
                rightAction: { showDebugLog = true }
            )

            GeometryReader { proxy in
                ScrollView {
                    VStack(spacing: 0) {
                        Spacer(minLength: 0)
                            .frame(height: max(24, proxy.size.height * 0.18))

                        heroSection

                        Spacer(minLength: max(40, proxy.size.height * 0.14))

                        composerSection

                        Text("UCLH-approved guidelines - Apply clinical judgement")
                            .font(.system(size: 10.5, weight: .regular))
                            .foregroundStyle(Color.ngMuted.opacity(0.4))
                            .multilineTextAlignment(.center)
                            .padding(.top, 14)
                            .padding(.bottom, 20)
                    }
                    .frame(minHeight: proxy.size.height, alignment: .top)
                    .padding(.horizontal, 16)
                    .contentShape(Rectangle())
                    .onTapGesture {
                        dismissKeyboard()
                    }
                }
                .scrollDismissesKeyboard(.interactively)
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
        .toolbar {
            ToolbarItemGroup(placement: .keyboard) {
                Spacer()
                Button("Done") {
                    dismissKeyboard()
                }
            }
        }
        .fullScreenCover(isPresented: $showDebugLog) {
            DebugLogView()
        }
    }

    private var heroSection: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 6) {
                Circle()
                    .fill(Color.ngAccent)
                    .frame(width: 6, height: 6)
                Text("Clinical Guideline Retrieval")
                    .font(.system(size: 11, weight: .regular))
                    .foregroundStyle(Color.ngAccent)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 4)
            .background(Color.ngAccent.opacity(0.1), in: Capsule())
            .overlay(
                Capsule().stroke(Color.ngBorderAccent, lineWidth: 1)
            )

            Text("Ask anything about\nUCLH neonatal care.")
                .font(.system(size: 28, weight: .ultraLight))
                .tracking(-0.7)
                .foregroundStyle(Color.ngText)
                .lineSpacing(4)
                .lineLimit(nil)
                .multilineTextAlignment(.leading)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
                .layoutPriority(2)
                .padding(.top, 14)

            Text("Answers sourced directly from UCLH-approved guidelines, with citations and page references.")
                .font(.system(size: 13, weight: .light))
                .foregroundStyle(Color.ngMuted)
                .lineSpacing(6)
                .lineLimit(nil)
                .multilineTextAlignment(.leading)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
                .layoutPriority(1)
                .padding(.top, 12)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.vertical, 8)
    }

    private var composerSection: some View {
        NGCard {
            VStack(alignment: .leading, spacing: 10) {
                HStack(alignment: .bottom, spacing: 10) {
                    TextField("Ask a clinical question...", text: $query, axis: .vertical)
                        .lineLimit(3, reservesSpace: true)
                        .font(.system(size: 15, weight: .light))
                        .foregroundStyle(Color.ngText)
                        .submitLabel(.send)
                        .focused($isQueryFocused)
                        .onSubmit(submit)

                    Button(action: submit) {
                        Image(systemName: "arrow.up")
                            .font(.system(size: 13, weight: .semibold))
                            .foregroundStyle(query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? Color.ngMuted : Color.white)
                            .frame(width: 36, height: 36)
                            .background(
                                query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                                ? Color.white.opacity(0.04)
                                : Color.ngAccent,
                                in: RoundedRectangle(cornerRadius: 12)
                            )
                            .shadow(
                                color: query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                                ? .clear
                                : Color.ngAccent.opacity(0.12),
                                radius: 10,
                                x: 0,
                                y: 0
                            )
                    }
                    .buttonStyle(.plain)
                    .disabled(query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }

                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 6) {
                        ForEach(suggestions, id: \.self) { suggestion in
                            Button(action: { query = suggestion }) {
                                Text(suggestion)
                                    .font(.system(size: 11, weight: .regular))
                                    .foregroundStyle(Color.ngMuted)
                                    .padding(.horizontal, 10)
                                    .padding(.vertical, 4)
                                    .background(Color.white.opacity(0.04), in: Capsule())
                                    .overlay(
                                        Capsule().stroke(Color.ngBorder, lineWidth: 1)
                                    )
                            }
                            .buttonStyle(.plain)
                        }
                    }
                    .padding(.horizontal, 1)
                }
            }
            .padding(14)
        }
    }

    private func submit() {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        dismissKeyboard()
        onSubmit(trimmed)
    }

    private func dismissKeyboard() {
        isQueryFocused = false
        UIApplication.shared.sendAction(
            #selector(UIResponder.resignFirstResponder),
            to: nil,
            from: nil,
            for: nil
        )
    }
}
