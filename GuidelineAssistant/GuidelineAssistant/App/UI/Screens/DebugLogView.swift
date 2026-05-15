import SwiftUI
import UIKit

struct DebugLogView: View {
    @State private var store = DebugLogStore.shared
    @State private var showShareSheet = false
    @State private var filterText = ""
    @State private var autoscroll = true

    private var filteredEntries: [DebugLogStore.LogEntry] {
        guard !filterText.isEmpty else { return store.entries }

        let lower = filterText.lowercased()
        return store.entries.filter {
            $0.message.lowercased().contains(lower) ||
            $0.category.lowercased().contains(lower)
        }
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                TextField("Filter", text: $filterText)
                    .textFieldStyle(.roundedBorder)
                    .padding(.horizontal)
                    .padding(.top, 8)

                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 6) {
                            ForEach(filteredEntries) { entry in
                                LogEntryRow(entry: entry)
                                    .id(entry.id)
                            }
                        }
                        .padding(.horizontal)
                        .padding(.vertical, 8)
                    }
                    .onChange(of: store.entries.count) { _, _ in
                        guard autoscroll, let last = filteredEntries.last else { return }
                        withAnimation {
                            proxy.scrollTo(last.id, anchor: .bottom)
                        }
                    }
                }

                Toggle("Autoscroll", isOn: $autoscroll)
                    .padding(.horizontal)
                    .padding(.bottom, 8)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
            .navigationTitle("Debug Logs")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Clear") {
                        store.clear()
                    }
                }

                ToolbarItem(placement: .topBarTrailing) {
                    Button("Share") {
                        showShareSheet = true
                    }
                }
            }
            .sheet(isPresented: $showShareSheet) {
                ShareSheet(items: [store.exportAsText()])
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
    }
}

private struct LogEntryRow: View {
    let entry: DebugLogStore.LogEntry

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 6) {
                Text(entry.timestamp.formatted(.dateTime.hour().minute().second()))
                    .foregroundStyle(.secondary)

                Text(entry.level)
                    .foregroundStyle(levelColor)
                    .fontWeight(.semibold)

                Text(entry.category)
                    .foregroundStyle(.tint)
            }
            .font(.caption2.monospaced())

            Text(entry.message)
                .font(.caption.monospaced())
                .textSelection(.enabled)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var levelColor: Color {
        switch entry.level {
        case "ERROR":
            return .red
        case "WARN":
            return .orange
        case "INFO":
            return .primary
        default:
            return .secondary
        }
    }
}

private struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }

    func updateUIViewController(_ controller: UIActivityViewController, context: Context) {
        _ = context
    }
}
