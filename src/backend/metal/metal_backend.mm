#include "backend/metal/metal_backend.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <mach/vm_page_size.h>

#include <algorithm>
#include <array>
#include <string>
#include <unordered_map>
#include <vector>

#include "backend/checks.h"
#include "backend/rope.h"
#include "core/error.h"
#include "kernels/metal/ggml_args.h"

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

void MetalBackend::embed(const Tensor& table, const Tensor& ids, Tensor& out) {
  OpScope scope(*impl_);
  expect_dtype(ids, DType::I32, "embed ids");
  // Same allow-list and checks as CpuBackend::embed.
  const char* kernel = nullptr;
  switch (table.dtype()) {
    case DType::F32: kernel = "kernel_get_rows_f32"; break;
    case DType::F16: kernel = "kernel_get_rows_f16"; break;
    case DType::BF16: kernel = "kernel_get_rows_bf16"; break;
    case DType::Q8_0: kernel = "kernel_get_rows_q8_0"; break;
    default: fail<InvalidArgument>("embed table must be F32, F16, BF16 or Q8_0, got {}", tie::name(table.dtype()));
  }
  const bool quantized = table.dtype() == DType::Q8_0;
  if (quantized && table.cols() % kQ8_0BlockElems != 0) {
    fail<InvalidArgument>("embed: Q8_0 table needs columns divisible by {}, got {}", kQ8_0BlockElems, table.cols());
  }
  expect_dtype(out, DType::F32, "embed output");
  const int64_t vocab = table.rows();
  const int64_t dim = table.cols();
  expect_shape(out, Shape{ids.numel(), dim}, "embed output");
  const int32_t* tokens = ids.data<int32_t>();  // unified memory: ids are checked on the host
  for (int64_t t = 0; t < ids.numel(); ++t) {
    if (tokens[t] < 0 || tokens[t] >= vocab) {
      fail<InvalidArgument>("token id {} outside vocabulary of {}", tokens[t], vocab);
    }
  }

  // ggml view: src0 = table [ne00 = dim, ne01 = vocab], src1 = ids [ne10 = n], dst [ne0 = dim, ne1 = n].
  ggml_metal_kargs_get_rows args{};
  args.ne00t = static_cast<int32_t>(quantized ? dim / 16 : dim);  // the Q8_0 kernel writes a float4x4 per thread
  args.ne00 = static_cast<int32_t>(dim);
  args.nb01 = table.row_bytes();
  args.nb02 = args.nb03 = table.nbytes();
  args.ne10 = static_cast<int32_t>(ids.numel());
  args.nb10 = sizeof(int32_t);
  args.nb11 = args.nb12 = static_cast<uint64_t>(ids.numel()) * sizeof(int32_t);
  args.nb1 = out.row_bytes();
  args.nb2 = args.nb3 = out.nbytes();

  if (out.numel() > 0) {
    id<MTLComputePipelineState> pso = impl_->pipeline(kernel);
    const NSUInteger threads = std::min<NSUInteger>(static_cast<NSUInteger>(args.ne00t), pso.maxTotalThreadsPerThreadgroup);
    const NSUInteger groups_per_row = (static_cast<NSUInteger>(args.ne00t) + threads - 1) / threads;
    impl_->bind_value(args, 0);
    impl_->bind(table, 1);
    impl_->bind(ids, 2);
    impl_->bind(out, 3);
    impl_->dispatch_groups(pso, MTLSizeMake(groups_per_row * static_cast<NSUInteger>(ids.numel()), 1, 1),
                           MTLSizeMake(threads, 1, 1));
  }
  scope.finish();
}

void MetalBackend::matmul(const Tensor& x, const Tensor& w, Tensor& out) {
  OpScope scope(*impl_);
  // Same checks as CpuBackend::matmul, plus K % 4 == 0: the kernels read 4-wide.
  expect_dtype(x, DType::F32, "matmul input");
  expect_dtype(out, DType::F32, "matmul output");
  const char* mv_kernel = nullptr;
  const char* mm_kernel = nullptr;
  switch (w.dtype()) {
    case DType::F32: mv_kernel = "kernel_mul_mv_f32_f32_4"; mm_kernel = "kernel_mul_mm_f32_f32"; break;
    case DType::F16: mv_kernel = "kernel_mul_mv_f16_f32_4"; mm_kernel = "kernel_mul_mm_f16_f32"; break;
    case DType::BF16: mv_kernel = "kernel_mul_mv_bf16_f32_4"; mm_kernel = "tie_mul_mm_bf16_f32"; break;
    case DType::Q8_0: mv_kernel = "kernel_mul_mv_q8_0_f32"; mm_kernel = "kernel_mul_mm_q8_0_f32"; break;
    default: fail<InvalidArgument>("matmul weight dtype {} unsupported", tie::name(w.dtype()));
  }
  const int64_t T = x.rows();
  const int64_t K = x.cols();
  const int64_t N = w.rows();
  if (w.cols() != K) fail<InvalidArgument>("matmul: input {} does not match weight {}", x.shape().str(), w.shape().str());
  if (w.dtype() == DType::Q8_0 && K % kQ8_0BlockElems != 0) {
    fail<InvalidArgument>("matmul: Q8_0 weight needs K divisible by {}, got {}", kQ8_0BlockElems, K);
  }
  expect_shape(out, Shape{T, N}, "matmul output");
  if (K % 4 != 0) fail<InvalidArgument>("Metal matmul needs K % 4 == 0, got {}", K);
  // llama.cpp's rule (ggml_metal_op_mul_mat_use_mm): the matmul kernel for K >= 64 and more than 8 rows.
  const bool use_mm = T > 8 && K >= 64;
  // The matvec kernels read weight rows in pairs, so an odd N would read one row past the
  // weights (possibly past the end of a memory-mapped file). The matmul path clamps its reads.
  if (!use_mm && N % 2 != 0) {
    fail<InvalidArgument>("Metal matvec kernels process weight rows in pairs: N must be even, got {}", N);
  }

  impl_->bind(w, 1);
  impl_->bind(x, 2);
  impl_->bind(out, 3);

  if (use_mm) {
    // Prefill: 64 (N) x 32 (T) output tiles; the grid is (T tiles, N tiles). Host values follow
    // ggml_metal_library_get_pipeline_mul_mm and ggml_metal_op_mul_mat (non-tensor path).
    ggml_metal_kargs_mul_mm args{};
    args.ne00 = static_cast<int32_t>(K);
    args.ne02 = 1;
    args.nb01 = w.row_bytes();
    args.nb02 = args.nb03 = w.nbytes();
    args.ne12 = 1;
    args.nb10 = sizeof(float);
    args.nb11 = x.row_bytes();
    args.nb12 = args.nb13 = x.nbytes();
    args.ne0 = static_cast<int32_t>(N);
    args.ne1 = static_cast<int32_t>(T);
    args.r2 = args.r3 = 1;
    const bool bc_inp = K % 32 != 0;
    const bool bc_out = N % 64 != 0 || T % 32 != 0;
    id<MTLComputePipelineState> pso = impl_->pipeline(
        mm_kernel, {{FC_MUL_MM + 0, true, bc_inp}, {FC_MUL_MM + 1, true, bc_out}, {FC_MUL_MM + 2, false, 1} /* ne12 */,
                    {FC_MUL_MM + 3, false, 1} /* ne13 */, {FC_MUL_MM + 4, false, 1} /* r2 */,
                    {FC_MUL_MM + 5, false, 1} /* r3 */});
    impl_->bind_value(args, 0);
    impl_->dispatch_groups(pso, MTLSizeMake(static_cast<NSUInteger>((T + 31) / 32), static_cast<NSUInteger>((N + 63) / 64), 1),
                           MTLSizeMake(32, 4, 1), bc_out ? 8192 : 4096 + 2048);
  } else {
    // Decode: each threadgroup reduces 2 weight rows against one activation row. Host values follow
    // ggml_metal_library_get_pipeline_mul_mv and ggml_metal_op_mul_mat_mv.
    const bool q8 = w.dtype() == DType::Q8_0;
    const int nsg = q8 ? N_SG_Q8_0 : static_cast<int>(std::min<int64_t>(4, (K + 127) / 128));
    constexpr int nr0 = 2;  // N_R0_Q8_0, and the only case the t_t_4 dispatcher instantiates
    ggml_metal_kargs_mul_mv args{};
    args.ne00 = static_cast<int32_t>(K);
    args.ne01 = static_cast<int32_t>(N);
    args.ne02 = 1;
    args.nb00 = static_cast<uint64_t>(traits(w.dtype()).block_bytes);  // GGML type size: one element, or one Q8_0 block
    args.nb01 = w.row_bytes();
    args.nb02 = args.nb03 = w.nbytes();
    args.ne10 = static_cast<int32_t>(K);
    args.ne11 = static_cast<int32_t>(T);
    args.ne12 = 1;
    args.nb10 = sizeof(float);
    args.nb11 = x.row_bytes();
    args.nb12 = args.nb13 = x.nbytes();
    args.ne0 = static_cast<int32_t>(N);
    args.ne1 = static_cast<int32_t>(T);
    args.nr0 = nr0;
    args.r2 = args.r3 = 1;
    id<MTLComputePipelineState> pso = impl_->pipeline(
        mv_kernel, {{FC_MUL_MV + 0, false, nsg}, {FC_MUL_MV + 2, false, 1} /* ne12 */, {FC_MUL_MV + 3, false, 1} /* r2 */,
                    {FC_MUL_MV + 4, false, 1} /* r3 */, {FC_MUL_MV + 5, true, 0} /* split */});
    impl_->bind_value(args, 0);
    impl_->dispatch_groups(pso, MTLSizeMake(static_cast<NSUInteger>((N + nr0 - 1) / nr0), static_cast<NSUInteger>(T), 1),
                           MTLSizeMake(32, static_cast<NSUInteger>(nsg), 1), 32 * sizeof(float) * nr0);
  }
  scope.finish();
}

void MetalBackend::rms_norm(const Tensor& x, const Tensor& weight, float eps, Tensor& out) {
  OpScope scope(*impl_);
  expect_dtype(x, DType::F32, "rms_norm input");
  expect_dtype(weight, DType::F32, "rms_norm weight");
  expect_shape(weight, Shape{x.cols()}, "rms_norm weight");
  expect_dtype(out, DType::F32, "rms_norm output");
  expect_shape(out, x.shape(), "rms_norm output");
  constexpr NSUInteger kReads = 4, kLoopedLimit = 4096, kSimd = 32;  // RMS_N_READS, RMS_LOOPED_LIMIT
  const auto axis = static_cast<uint32_t>(x.cols());
  const uint32_t w_stride = 1;

  if (x.numel() > 0) {
    // MLX dispatch: one threadgroup per row, sized to cover the row in reads of 4; rows longer than
    // the limit (or than one threadgroup can cover) use the looped kernel at the maximum size.
    id<MTLComputePipelineState> pso = impl_->pipeline("rms_f32");
    NSUInteger threads = kSimd * (((axis + kReads - 1) / kReads + kSimd - 1) / kSimd);
    if (axis > kLoopedLimit || threads > pso.maxTotalThreadsPerThreadgroup) {
      pso = impl_->pipeline("rms_looped_f32");
      threads = pso.maxTotalThreadsPerThreadgroup;
    }
    impl_->bind(x, 0);
    impl_->bind(weight, 1);
    impl_->bind(out, 2);
    impl_->bind_value(eps, 3);
    impl_->bind_value(axis, 4);
    impl_->bind_value(w_stride, 5);
    impl_->dispatch_groups(pso, MTLSizeMake(static_cast<NSUInteger>(x.rows()), 1, 1), MTLSizeMake(threads, 1, 1));
  }
  scope.finish();
}

void MetalBackend::rope_neox(Tensor& x, const Tensor& positions, int num_heads, int head_dim, float theta) {
  OpScope scope(*impl_);
  expect_dtype(x, DType::F32, "rope input");
  expect_dtype(positions, DType::I32, "rope positions");
  if (head_dim % 2 != 0 || head_dim > kMaxHeadDim) fail<InvalidArgument>("rope head_dim {} unsupported", head_dim);
  if (num_heads <= 0 || head_dim <= 0) fail<InvalidArgument>("rope needs positive num_heads and head_dim, got {} and {}", num_heads, head_dim);
  expect_shape(positions, Shape{x.rows()}, "rope positions");
  const int64_t T = positions.numel();
  expect_shape(x, Shape{T, static_cast<int64_t>(num_heads) * head_dim}, "rope input");

  // x viewed as ggml [ne0 = head_dim, ne1 = heads, ne2 = tokens]; byte strides; rotated in place.
  ggml_metal_kargs_rope args{};
  args.ne00 = args.ne0 = head_dim;
  args.ne01 = args.ne1 = num_heads;
  args.ne02 = args.ne2 = static_cast<int32_t>(T);
  args.ne03 = args.ne3 = 1;
  args.nb00 = args.nb0 = sizeof(float);
  args.nb01 = args.nb1 = static_cast<uint64_t>(head_dim) * sizeof(float);
  args.nb02 = args.nb2 = x.row_bytes();
  args.nb03 = args.nb3 = x.nbytes();
  args.n_dims = head_dim;
  args.n_offs = 0;
  args.n_ctx_orig = 0;
  args.freq_base = theta;
  args.freq_scale = 1.0f;
  args.ext_factor = 0.0f;  // plain RoPE: no YaRN
  args.attn_factor = 1.0f;
  args.beta_fast = 32.0f;
  args.beta_slow = 1.0f;
  args.src2 = true;  // tie's kernel reads inv_freq from the freq-factor slot
  args.inplace = true;
  std::array<float, kMaxHeadDim / 2> inv_freq{};  // same table as the CPU backend: identical angles
  rope_inv_freq(theta, head_dim, inv_freq.data());

  id<MTLComputePipelineState> pso = impl_->pipeline(
      "kernel_rope_neox_f32", {{FC_ROPE + 0, true, 0} /* imrope */, {FC_ROPE + 1, true, 0} /* is_back */});
  impl_->bind_value(args, 0);
  impl_->bind(x, 1);
  impl_->bind(positions, 2);
  impl_->bind_value(inv_freq, 3);  // 2 KiB, under setBytes' 4 KiB limit
  impl_->bind(x, 4);  // dst == src: in place
  impl_->dispatch_groups(pso, MTLSizeMake(static_cast<NSUInteger>(num_heads), static_cast<NSUInteger>(T), 1),
                         MTLSizeMake(std::min<NSUInteger>(1024, static_cast<NSUInteger>(head_dim)), 1, 1));
  scope.finish();
}

void MetalBackend::silu_mul(const Tensor& gate, const Tensor& up, Tensor& out) {
  OpScope scope(*impl_);
  expect_dtype(gate, DType::F32, "silu_mul gate");
  expect_dtype(up, DType::F32, "silu_mul up");
  expect_dtype(out, DType::F32, "silu_mul output");
  expect_shape(up, gate.shape(), "silu_mul up");
  expect_shape(out, gate.shape(), "silu_mul output");
  ggml_metal_kargs_glu args{};
  args.ne00 = args.ne10 = args.ne0 = static_cast<int32_t>(gate.cols());
  args.nb01 = args.nb11 = args.nb1 = gate.row_bytes();
  args.i00 = args.i10 = 0;  // gate and up are separate tensors
  args.alpha = 0.0f;
  args.limit = 0.0f;

  if (gate.numel() > 0) {
    id<MTLComputePipelineState> pso = impl_->pipeline("kernel_swiglu_f32");
    // llama.cpp: max(1, min(max threads, ne00 / 2)) threads stride over each row.
    const NSUInteger threads = std::max<NSUInteger>(
        1, std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup, static_cast<NSUInteger>(gate.cols()) / 2));
    impl_->bind_value(args, 0);
    impl_->bind(gate, 1);
    impl_->bind(up, 2);
    impl_->bind(out, 3);
    impl_->dispatch_groups(pso, MTLSizeMake(static_cast<NSUInteger>(gate.rows()), 1, 1), MTLSizeMake(threads, 1, 1));
  }
  scope.finish();
}

void MetalBackend::kv_write(const Tensor&, const Tensor&, const Tensor&, Tensor&, Tensor&) {
  OpScope scope(*impl_);
  fail<UnsupportedError>("Metal kv_write lands in Task 17");
}

void MetalBackend::paged_attention(const Tensor&, const Tensor&, const Tensor&, const AttentionMetadata&, int, float,
                                   Tensor&) {
  OpScope scope(*impl_);
  fail<UnsupportedError>("Metal paged_attention lands in Task 17");
}

}  // namespace tie
