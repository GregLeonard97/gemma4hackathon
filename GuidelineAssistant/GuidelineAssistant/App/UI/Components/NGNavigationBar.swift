import SwiftUI

struct NGNavigationBar: View {
    let title: String
    let subtitle: String?
    let backAction: (() -> Void)?
    let rightLabel: String?
    let rightSystemImage: String?
    let rightAction: (() -> Void)?

    init(
        title: String,
        subtitle: String? = nil,
        backAction: (() -> Void)? = nil,
        rightLabel: String? = nil,
        rightSystemImage: String? = nil,
        rightAction: (() -> Void)? = nil
    ) {
        self.title = title
        self.subtitle = subtitle
        self.backAction = backAction
        self.rightLabel = rightLabel
        self.rightSystemImage = rightSystemImage
        self.rightAction = rightAction
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack(alignment: .center, spacing: 12) {
                if let backAction {
                    Button(action: backAction) {
                        HStack(spacing: 4) {
                            Image(systemName: "chevron.left")
                                .font(.system(size: 13, weight: .semibold))
                            Text("Back")
                                .font(.system(size: 15, weight: .regular))
                        }
                        .foregroundStyle(Color.ngAccent)
                    }
                    .accessibilityLabel("Back")
                }

                VStack(alignment: .leading, spacing: 1) {
                    Text(title)
                        .font(.system(size: 17, weight: .semibold))
                        .foregroundStyle(Color.ngText)
                    if let subtitle {
                        Text(subtitle)
                            .font(.system(size: 11, weight: .regular))
                            .foregroundStyle(Color.ngAccent)
                    }
                }

                Spacer()

                if let rightLabel {
                    Text(rightLabel)
                        .font(.system(size: 11, weight: .regular))
                        .foregroundStyle(Color.ngAccent)
                } else if let rightSystemImage, let rightAction {
                    Button(action: rightAction) {
                        Image(systemName: rightSystemImage)
                            .font(.system(size: 15, weight: .regular))
                            .foregroundStyle(Color.ngMuted)
                            .frame(width: 32, height: 32)
                            .background(Color.white.opacity(0.06), in: Circle())
                            .overlay(Circle().stroke(Color.white.opacity(0.1), lineWidth: 1))
                    }
                    .buttonStyle(.plain)
                } else if backAction == nil {
                    Button(action: {}) {
                        Image(systemName: "info.circle")
                            .font(.system(size: 15, weight: .regular))
                            .foregroundStyle(Color.ngMuted)
                            .frame(width: 32, height: 32)
                            .background(Color.white.opacity(0.06), in: Circle())
                            .overlay(Circle().stroke(Color.white.opacity(0.1), lineWidth: 1))
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("Information")
                }
            }
            .padding(.horizontal, 16)
            .padding(.top, 10)
            .padding(.bottom, 10)
        }
        .background(Color.ngNavBg)
        .background(.ultraThinMaterial.opacity(0.2))
        .overlay(alignment: .bottom) {
            Rectangle()
                .fill(Color.ngBorder)
                .frame(height: 1)
        }
    }
}
