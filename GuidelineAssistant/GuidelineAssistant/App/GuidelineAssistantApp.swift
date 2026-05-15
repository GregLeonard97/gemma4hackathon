import SwiftUI

@main
struct GuidelineAssistantApp: App {
    @State private var appState = AppState()

    var body: some Scene {
        WindowGroup {
            NeoGuideRootView(appState: appState)
        }
    }
}
