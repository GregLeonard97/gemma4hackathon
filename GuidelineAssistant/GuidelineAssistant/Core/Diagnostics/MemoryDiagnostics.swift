import Foundation
import Darwin.Mach
import os

enum MemoryDiagnostics {
    /// Returns one-line memory state including phys footprint and remaining headroom.
    static func report() -> String {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info>.size / MemoryLayout<integer_t>.size
        )

        let result = withUnsafeMutablePointer(to: &info) { infoPtr in
            infoPtr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPtr in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), intPtr, &count)
            }
        }

        guard result == KERN_SUCCESS else {
            return "Memory info unavailable: kern_return \(result)"
        }

        let physMB = Double(info.phys_footprint) / 1_000_000
        let remainingMB = Double(info.limit_bytes_remaining) / 1_000_000
        let totalLimitMB = physMB + remainingMB

        return "phys=\(String(format: "%.0f", physMB))MB, remaining=\(String(format: "%.0f", remainingMB))MB, total_limit~\(String(format: "%.0f", totalLimitMB))MB"
    }

    /// Writes memory diagnostics to both os.Logger and the in-app persistent log store.
    static func log(stage: String, logger: Logger? = nil) {
        let message = "[\(stage)] \(report())"
        logger?.info("\(message, privacy: .public)")
        DebugLogStore.shared.log(level: "INFO", category: "Memory", message: message)
    }
}
