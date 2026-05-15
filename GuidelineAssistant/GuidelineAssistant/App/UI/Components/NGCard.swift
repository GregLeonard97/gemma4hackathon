import SwiftUI

struct NGCard<Content: View>: View {
    private let content: Content

    init(@ViewBuilder content: () -> Content) {
        self.content = content()
    }

    var body: some View {
        content
            .background(Color.ngSurface)
            .overlay(
                RoundedRectangle(cornerRadius: NGTheme.cornerRadiusLarge)
                    .stroke(Color.ngBorder, lineWidth: 1)
            )
            .clipShape(RoundedRectangle(cornerRadius: NGTheme.cornerRadiusLarge))
            .shadow(color: Color.ngAccent.opacity(0.15), radius: 25, x: 0, y: 10)
    }
}
