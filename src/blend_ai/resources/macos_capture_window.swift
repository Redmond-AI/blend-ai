import AppKit
import CoreGraphics
import Foundation
import ImageIO
import ScreenCaptureKit
import UniformTypeIdentifiers

struct Bounds: Encodable {
    let x: Int
    let y: Int
    let width: Int
    let height: Int
}

struct CaptureResult: Encodable {
    let windowID: Int
    let ownerPID: Int
    let width: Int
    let height: Int
    let bounds: Bounds
}

func fail(_ message: String, code: Int32 = 1) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(code)
}

guard CommandLine.arguments.count == 7,
      let requestedPID = Int(CommandLine.arguments[1]),
      let expectedWidth = Int(CommandLine.arguments[2]),
      let expectedHeight = Int(CommandLine.arguments[3]),
      let expectedX = Int(CommandLine.arguments[4]),
      let expectedY = Int(CommandLine.arguments[5]) else {
    fail(
        "usage: macos_capture_window.swift PID WIDTH HEIGHT X Y OUTPUT.png",
        code: 2
    )
}

let outputURL = URL(fileURLWithPath: CommandLine.arguments[6])
guard CGPreflightScreenCaptureAccess() else {
    fail(
        "Screen recording permission is required. Enable Codex in System Settings " +
        "> Privacy & Security > Screen & System Audio Recording, then restart Codex."
    )
}
let options: CGWindowListOption = [.optionOnScreenOnly, .excludeDesktopElements]
guard let rawWindows = CGWindowListCopyWindowInfo(options, kCGNullWindowID)
        as? [[String: Any]] else {
    fail("CoreGraphics did not return an on-screen window list")
}

struct Candidate {
    let id: CGWindowID
    let pid: Int
    let layer: Int
    let bounds: Bounds
    let score: Int
}

var candidates: [Candidate] = []
let maximumWidthDelta = 4
let maximumHeightChrome = 128
let maximumXDelta = 8
let maximumYChrome = 128
for entry in rawWindows {
    guard let pid = entry[kCGWindowOwnerPID as String] as? Int,
          pid == requestedPID,
          let number = entry[kCGWindowNumber as String] as? Int,
          let rawBounds = entry[kCGWindowBounds as String] as? [String: Any] else {
        continue
    }
    let layer = (entry[kCGWindowLayer as String] as? Int) ?? 0
    let x = Int((rawBounds["X"] as? Double) ?? 0)
    let y = Int((rawBounds["Y"] as? Double) ?? 0)
    let width = Int((rawBounds["Width"] as? Double) ?? 0)
    let height = Int((rawBounds["Height"] as? Double) ?? 0)
    guard width > 0, height > 0, layer == 0 else { continue }
    let widthDelta = abs(width - expectedWidth)
    let heightDelta = height - expectedHeight
    let xDelta = abs(x - expectedX)
    let yDelta = abs(y - expectedY)
    // Blender RNA reports the content dimensions and a decoration-free origin;
    // CoreGraphics includes the macOS title bar. Only that small vertical
    // difference is allowed. A nearest-but-distant Blender window is never a
    // valid substitute for a hidden/minimized AI Preview window.
    guard widthDelta <= maximumWidthDelta,
          heightDelta >= 0,
          heightDelta <= maximumHeightChrome,
          xDelta <= maximumXDelta,
          yDelta <= maximumYChrome else {
        continue
    }
    let layerPenalty = layer == 0 ? 0 : 1_000_000_000
    // Blender's RNA height excludes title-bar chrome while CGWindow bounds
    // include it. Width and position still uniquely identify multiple main
    // windows in the same Blender process; height is a lower-weight tiebreak.
    let dimensionDelta = abs(width - expectedWidth) * 4 + abs(height - expectedHeight)
    let positionDelta = abs(x - expectedX) * 2 + abs(y - expectedY)
    let score = layerPenalty + dimensionDelta + positionDelta
    candidates.append(
        Candidate(
            id: CGWindowID(number),
            pid: pid,
            layer: layer,
            bounds: Bounds(x: x, y: y, width: width, height: height),
            score: score
        )
    )
}

let orderedCandidates = candidates.sorted(by: {
    if $0.score != $1.score { return $0.score < $1.score }
    return $0.bounds.width * $0.bounds.height > $1.bounds.width * $1.bounds.height
})
guard let selected = orderedCandidates.first else {
    fail(
        "The requested Blender preview window is not visible at the expected " +
        "position and dimensions for PID \(requestedPID)"
    )
}
if orderedCandidates.count > 1 && orderedCandidates[1].score == selected.score {
    fail("Multiple Blender windows match the requested capture target")
}

let shareableContent: SCShareableContent
do {
    shareableContent = try await SCShareableContent.excludingDesktopWindows(
        false,
        onScreenWindowsOnly: true
    )
} catch {
    fail("ScreenCaptureKit could not list shareable windows: \(error)")
}
guard let shareableWindow = shareableContent.windows.first(where: {
    $0.windowID == selected.id
}) else {
    fail("Blender window \(selected.id) is not available to ScreenCaptureKit")
}

let filter = SCContentFilter(desktopIndependentWindow: shareableWindow)
let configuration = SCStreamConfiguration()
configuration.width = selected.bounds.width
configuration.height = selected.bounds.height
configuration.showsCursor = false
configuration.captureResolution = .best
configuration.ignoreShadowsSingleWindow = true
configuration.ignoreGlobalClipSingleWindow = true

let image: CGImage
do {
    image = try await SCScreenshotManager.captureImage(
        contentFilter: filter,
        configuration: configuration
    )
} catch {
    fail("ScreenCaptureKit could not capture Blender window \(selected.id): \(error)")
}

guard let destination = CGImageDestinationCreateWithURL(
    outputURL as CFURL,
    UTType.png.identifier as CFString,
    1,
    nil
) else {
    fail("Could not create PNG destination")
}
CGImageDestinationAddImage(destination, image, nil)
guard CGImageDestinationFinalize(destination) else {
    fail("Could not write PNG destination")
}

let result = CaptureResult(
    windowID: Int(selected.id),
    ownerPID: selected.pid,
    width: image.width,
    height: image.height,
    bounds: selected.bounds
)
let encoder = JSONEncoder()
encoder.outputFormatting = [.sortedKeys]
guard let data = try? encoder.encode(result),
      let json = String(data: data, encoding: .utf8) else {
    fail("Could not encode capture metadata")
}
print(json)
