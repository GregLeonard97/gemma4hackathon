import SwiftUI

enum NGTheme {
    static let cornerRadiusLarge: CGFloat = 22
    static let cornerRadiusMedium: CGFloat = 12
}

extension Color {
    static let ngBackground = Color(hex: "#030108")
    static let ngSurface = Color(hex: "#0A0A10")
    static let ngNavBg = Color(hex: "#030108").opacity(0.85)
    static let ngAccent = Color(hex: "#2DD4BF")
    static let ngAccent2 = Color(hex: "#5EEAD4")
    static let ngText = Color.white
    static let ngMuted = Color(hex: "#9CA3AF")
    static let ngBorder = Color.white.opacity(0.08)
    static let ngBorderAccent = Color(hex: "#2DD4BF").opacity(0.3)
    static let ngSourceIcon = Color(hex: "#2DD4BF").opacity(0.1)
    static let ngNoteBg = Color(hex: "#2DD4BF").opacity(0.07)
    static let ngHighlight = Color(hex: "#2DD4BF").opacity(0.08)
}

extension Color {
    init(hex: String) {
        let sanitized = hex
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "#", with: "")

        var rgb: UInt64 = 0
        Scanner(string: sanitized).scanHexInt64(&rgb)

        let red = Double((rgb >> 16) & 0xFF) / 255.0
        let green = Double((rgb >> 8) & 0xFF) / 255.0
        let blue = Double(rgb & 0xFF) / 255.0

        self.init(red: red, green: green, blue: blue)
    }
}
