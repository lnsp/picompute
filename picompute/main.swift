import Metal

let NUM_OF_STEPS = 10000001
let SIZE_OF_BLOCKS = 10000
let NUM_OF_BLOCKS = (NUM_OF_STEPS + SIZE_OF_BLOCKS - 1) / SIZE_OF_BLOCKS

// Allocate device, command queue and function
let device = MTLCreateSystemDefaultDevice()!
let commandQueue = device.makeCommandQueue()!
let library = device.makeDefaultLibrary()!
let calculatePi = library.makeFunction(name: "calculatePi")!
let pipeline = try! device.makeComputePipelineState(function: calculatePi)

// Setup offset values
var from: CFloat = -1.0
var to: CFloat = 1.0
var numOfSteps: CInt = CInt(NUM_OF_STEPS)
var stepSize: CFloat = (to - from) / CFloat(numOfSteps)
var sizeOfBlocks: CInt = CInt(SIZE_OF_BLOCKS)
var numOfBlocks: CInt = CInt(NUM_OF_BLOCKS)

// Allocate buffers
var x = [CFloat](repeating: 0.0, count: NUM_OF_STEPS)
for index in 0..<NUM_OF_STEPS {
    x[index] = CFloat(index) * stepSize + from
}
let xBuffer = device.makeBuffer(bytes: &x, length: MemoryLayout<CFloat>.stride * NUM_OF_STEPS, options: [])!
let yBuffer = device.makeBuffer(length: MemoryLayout<CFloat>.stride * NUM_OF_BLOCKS, options: [])!
let y = UnsafeBufferPointer<CFloat>(start: UnsafePointer(yBuffer.contents().assumingMemoryBound(to: CFloat.self)), count: NUM_OF_BLOCKS)
print("Allocated buffers: \(xBuffer.length + yBuffer.length) bytes")

// Setup thread groups
let threadgroupsPerGrid = MTLSize(width: (NUM_OF_STEPS + pipeline.threadExecutionWidth - 1) / pipeline.threadExecutionWidth, height: 1, depth: 1)
let threadsPerThreadgroup = MTLSize(width: pipeline.threadExecutionWidth, height: 1, depth: 1)

let commands = commandQueue.makeCommandBuffer()!
let encoder = commands.makeComputeCommandEncoder()!

// Encode values
encoder.setComputePipelineState(pipeline)
encoder.setBuffer(xBuffer, offset: 0, index: 0)
encoder.setBytes(&numOfSteps, length: MemoryLayout<CInt>.stride, index: 1)
encoder.setBuffer(yBuffer, offset: 0, index: 2)
encoder.setBytes(&sizeOfBlocks, length: MemoryLayout<CInt>.stride, index: 3)

encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
encoder.endEncoding()

// Time execution, run command
let startTime = Date()
commands.commit()
commands.waitUntilCompleted()
let endTime = Date()
let elapsedTime = endTime.timeIntervalSince(startTime)
print("Elapsed time: \(elapsedTime) seconds")

// Sum up results
var sum: CFloat = 0.0
for index in 0..<NUM_OF_BLOCKS {
    sum += y[index]
}

// Calculate Pi
let pi = 2.0 * sum * stepSize

print("Computed value of Pi: \(pi)")
