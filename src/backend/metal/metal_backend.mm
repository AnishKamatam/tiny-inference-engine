#include "backend/metal/metal_backend.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <mach/vm_page_size.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "backend/checks.h"
#include "core/error.h"

namespace tie {

extern const char kMetalSource[];

namespace {

// A Buffer backed by an MTLBuffer. `base_offset` is where data() sits inside the
// MTLBuffer: non-zero when wrapping memory that does not start on a page boundary.
class MetalBuffer final : public Buffer {
 public:
  MetalBuffer(id<MTLBuffer> buffer, void* data, size_t size, size_t base_offset)
      : Buffer(data, size), buffer_(buffer), base_offset_(base_offset) {}
  id<MTLBuffer> mtl() const { return buffer_; }
  size_t base_offset() const { return base_offset_; }

 private:
  id<MTLBuffer> buffer_;
  size_t base_offset_;
};

// A function-constant value for pipeline specialization.
struct FunctionConstant {
  int index;
  bool is_bool;  // otherwise a 16-bit integer
  int value;
};

std::string ns_error(NSError* error) {
  return error != nil ? std::string(error.localizedDescription.UTF8String) : std::string("unknown error");
}

}  // namespace

struct MetalBackend::Impl {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLLibrary> library = nil;
  std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;

  id<MTLCommandBuffer> command_buffer = nil;
  id<MTLComputeCommandEncoder> encoder = nil;

  // Pipelines are created on first use, keyed by kernel name plus constant values.
  id<MTLComputePipelineState> pipeline(const std::string& kernel, const std::vector<FunctionConstant>& constants = {}) {
    std::string key = kernel;
    for (const FunctionConstant& c : constants) key += "|" + std::to_string(c.index) + (c.is_bool ? "b=" : "s=") + std::to_string(c.value);
    if (const auto it = pipelines.find(key); it != pipelines.end()) return it->second;

    NSError* error = nil;
    NSString* fn_name = [NSString stringWithUTF8String:kernel.c_str()];
    id<MTLFunction> fn = nil;
    if (constants.empty()) {
      fn = [library newFunctionWithName:fn_name];
    } else {
      MTLFunctionConstantValues* values = [MTLFunctionConstantValues new];
      for (const FunctionConstant& c : constants) {
        if (c.is_bool) {
          const bool v = c.value != 0;
          [values setConstantValue:&v type:MTLDataTypeBool atIndex:static_cast<NSUInteger>(c.index)];
        } else {
          const int16_t v = static_cast<int16_t>(c.value);
          [values setConstantValue:&v type:MTLDataTypeShort atIndex:static_cast<NSUInteger>(c.index)];
        }
      }
      fn = [library newFunctionWithName:fn_name constantValues:values error:&error];
    }
    if (fn == nil) fail<Error>("Metal kernel '{}' not found or not specializable: {}", kernel, ns_error(error));
    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&error];
    if (pso == nil) fail<Error>("Metal pipeline '{}' failed: {}", kernel, ns_error(error));
    pipelines.emplace(key, pso);
    return pso;
  }

  void open() {
    command_buffer = [queue commandBuffer];
    encoder = [command_buffer computeCommandEncoder];  // serial dispatch: each op sees the previous one's writes
  }

  void close() {
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    const bool failed = command_buffer.status == MTLCommandBufferStatusError;
    NSError* error = command_buffer.error;
    encoder = nil;
    command_buffer = nil;
    if (failed) fail<Error>("Metal command buffer failed: {}", ns_error(error));
  }

  // Abandons the open step without submitting it. Never throws.
  void discard() {
    if (encoder != nil) [encoder endEncoding];
    encoder = nil;
    command_buffer = nil;
  }

  void bind(const Tensor& t, int index) const {
    const auto* buffer = dynamic_cast<const MetalBuffer*>(t.buffer());
    if (buffer == nullptr) {
      fail<InvalidArgument>("tensor {} is not in Metal memory; allocate it with MetalBackend::alloc or wrap",
                            t.shape().str());
    }
    [encoder setBuffer:buffer->mtl() offset:buffer->base_offset() + t.offset() atIndex:static_cast<NSUInteger>(index)];
  }

  template <typename T>
  void bind_value(const T& value, int index) const {
    [encoder setBytes:&value length:sizeof(T) atIndex:static_cast<NSUInteger>(index)];
  }

  // One thread per element of a 1-D range.
  void dispatch_threads(id<MTLComputePipelineState> pso, size_t n) const {
    if (n == 0) return;
    [encoder setComputePipelineState:pso];
    const NSUInteger width = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup, 256);
    [encoder dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
  }

  // Explicit threadgroup grid, optional threadgroup memory at index 0.
  void dispatch_groups(id<MTLComputePipelineState> pso, MTLSize groups, MTLSize threads, size_t smem = 0) const {
    if (groups.width == 0 || groups.height == 0 || groups.depth == 0) return;
    [encoder setComputePipelineState:pso];
    if (smem > 0) [encoder setThreadgroupMemoryLength:smem atIndex:0];
    [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
  }
};

// Wraps one op. Pattern for every op: OpScope first, then validate, then encode, then
//   scope.finish();
// If an op throws before finish(), the destructor discards the whole current step (implicit
// or explicit), so a failed op never leaves a half-encoded command buffer behind.
// Construct first in every op, before validation, so any throw discards the step.
class MetalBackend::OpScope {
 public:
  explicit OpScope(Impl& impl) : impl_(impl) {
    if (impl_.encoder == nil) {
      impl_.open();
      opened_ = true;
    }
  }
  OpScope(const OpScope&) = delete;
  OpScope& operator=(const OpScope&) = delete;
  ~OpScope() {
    if (!finished_) impl_.discard();
  }
  void finish() {
    finished_ = true;
    if (opened_) impl_.close();
  }

 private:
  Impl& impl_;
  bool opened_ = false;
  bool finished_ = false;
};

MetalBackend::MetalBackend() : impl_(std::make_unique<Impl>()) {
  impl_->device = MTLCreateSystemDefaultDevice();
  if (impl_->device == nil) fail<UnsupportedError>("no Metal device available; use --device cpu");
  impl_->queue = [impl_->device newCommandQueue];

  MTLCompileOptions* options = [MTLCompileOptions new];
  options.languageVersion = MTLLanguageVersion3_1;  // bfloat support
  options.preprocessorMacros = @{@"GGML_METAL_HAS_BF16" : @1};
  NSError* error = nil;
  impl_->library = [impl_->device newLibraryWithSource:[NSString stringWithUTF8String:kMetalSource]
                                               options:options
                                                 error:&error];
  if (impl_->library == nil) fail<Error>("Metal shader compilation failed: {}", ns_error(error));
}

MetalBackend::~MetalBackend() { impl_->discard(); }

std::unique_ptr<Buffer> MetalBackend::alloc(size_t bytes) {
  id<MTLBuffer> buffer = [impl_->device newBufferWithLength:std::max<size_t>(bytes, 1)
                                                    options:MTLResourceStorageModeShared];
  if (buffer == nil) fail<CapacityError>("Metal could not allocate {} bytes", bytes);
  return std::make_unique<MetalBuffer>(buffer, buffer.contents, bytes, 0);
}

std::unique_ptr<Buffer> MetalBackend::wrap(const void* data, size_t bytes) {
  // newBufferWithBytesNoCopy needs a page-aligned start and length, so wrap the
  // enclosing whole pages and remember where `data` sits inside them.
  if (bytes == 0) fail<InvalidArgument>("Metal cannot wrap zero bytes");
  const auto addr = reinterpret_cast<uintptr_t>(data);
  const uintptr_t page = vm_page_size;
  const uintptr_t start = addr & ~(page - 1);
  const uintptr_t end = (addr + bytes + page - 1) & ~(page - 1);
  id<MTLBuffer> buffer = [impl_->device newBufferWithBytesNoCopy:reinterpret_cast<void*>(start)
                                                          length:end - start
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];
  if (buffer == nil) fail<CapacityError>("Metal could not wrap {} bytes at {}", bytes, data);
  return std::make_unique<MetalBuffer>(buffer, const_cast<void*>(data), bytes, addr - start);
}

void MetalBackend::begin_step() {
  if (impl_->encoder != nil) fail<InvalidArgument>("begin_step called while a step is already open");
  impl_->open();
}

void MetalBackend::end_step() {
  if (impl_->encoder == nil) fail<InvalidArgument>("end_step called without begin_step");
  impl_->close();
}

void MetalBackend::add(const Tensor& a, const Tensor& b, Tensor& out) {
  OpScope scope(*impl_);
  expect_dtype(a, DType::F32, "add input");
  expect_dtype(b, DType::F32, "add input");
  expect_dtype(out, DType::F32, "add output");
  expect_shape(b, a.shape(), "add input");
  expect_shape(out, a.shape(), "add output");
  impl_->bind(a, 0);
  impl_->bind(b, 1);
  impl_->bind(out, 2);
  impl_->bind_value(static_cast<uint32_t>(a.numel()), 3);
  impl_->dispatch_threads(impl_->pipeline("tie_add_f32"), static_cast<size_t>(a.numel()));
  scope.finish();
}

void MetalBackend::gather_rows(const Tensor& x, const Tensor& rows, Tensor& out) {
  OpScope scope(*impl_);
  expect_dtype(x, DType::F32, "gather_rows input");
  expect_dtype(rows, DType::I32, "gather_rows indices");
  expect_dtype(out, DType::F32, "gather_rows output");
  expect_shape(out, Shape{rows.numel(), x.cols()}, "gather_rows output");
  const int32_t* r = rows.data<int32_t>();  // unified memory: indices are checked on the host
  for (int64_t i = 0; i < rows.numel(); ++i) {
    if (r[i] < 0 || r[i] >= x.rows()) fail<InvalidArgument>("gather row {} outside [0, {})", r[i], x.rows());
  }
  impl_->bind(x, 0);
  impl_->bind(rows, 1);
  impl_->bind(out, 2);
  impl_->bind_value(static_cast<uint32_t>(x.cols()), 3);
  impl_->bind_value(static_cast<uint32_t>(out.numel()), 4);
  impl_->dispatch_threads(impl_->pipeline("tie_gather_rows_f32"), static_cast<size_t>(out.numel()));
  scope.finish();
}

// Ported kernels arrive in Tasks 13 (embed, rms_norm, rope_neox, silu_mul) and 14 (matmul).
void MetalBackend::embed(const Tensor&, const Tensor&, Tensor&){
  OpScope scope(*impl_);
  fail<UnsupportedError>("Metal embed lands in Task 13");
}
void MetalBackend::matmul(const Tensor&, const Tensor&, Tensor&){
  OpScope scope(*impl_);
  fail<UnsupportedError>("Metal matmul lands in Task 14");
}
void MetalBackend::rms_norm(const Tensor&, const Tensor&, float, Tensor&){
  OpScope scope(*impl_);
  fail<UnsupportedError>("Metal rms_norm lands in Task 13");
}
void MetalBackend::rope_neox(Tensor&, const Tensor&, int, int, float){
  OpScope scope(*impl_);
  fail<UnsupportedError>("Metal rope_neox lands in Task 13");
}
void MetalBackend::silu_mul(const Tensor&, const Tensor&, Tensor&){
  OpScope scope(*impl_);
  fail<UnsupportedError>("Metal silu_mul lands in Task 13");
}

}  // namespace tie
