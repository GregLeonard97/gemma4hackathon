import SwiftUI

struct ParticleBackground: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var phase: Double = 0

    private let timer = Timer.publish(every: 0.05, on: .main, in: .common).autoconnect()

    var body: some View {
        Canvas { context, size in
            let spacing: Double = 28
            let columns = Int(size.width / spacing) + 1
            let rows = Int(size.height / spacing) + 1

            for row in 0..<rows {
                for column in 0..<columns {
                    let x = Double(column) * spacing + spacing / 2
                    let y = Double(row) * spacing + spacing / 2
                    let distance = hypot(x - size.width / 2, y - size.height / 2) / max(size.width, size.height)

                    let animatedPhase = reduceMotion ? 0 : phase
                    let pulse = 0.5 + 0.5 * sin(animatedPhase + distance * 8)
                    let alpha = max(0, (0.06 + 0.14 * pulse) * (1 - distance * 1.2))

                    let dotRect = CGRect(x: x - 1.2, y: y - 1.2, width: 2.4, height: 2.4)
                    context.fill(Path(ellipseIn: dotRect), with: .color(Color.ngAccent.opacity(alpha)))
                }
            }
        }
        .ignoresSafeArea()
        .opacity(0.18)
        .onReceive(timer) { _ in
            guard !reduceMotion else { return }
            phase += 0.035
        }
    }
}
