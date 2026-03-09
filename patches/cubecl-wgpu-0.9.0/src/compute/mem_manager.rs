use crate::{WgpuResource, WgpuStorage};
use cubecl_common::{backtrace::BackTrace, stream_id::StreamId, stub::Arc};
use cubecl_core::{
    MemoryConfiguration,
    server::{Binding, Handle, IoError},
};
use cubecl_ir::MemoryDeviceProperties;
use cubecl_runtime::{
    logging::ServerLogger,
    memory_management::{
        MemoryAllocationMode, MemoryHandle, MemoryManagement, MemoryManagementOptions,
        SliceBinding,
    },
    storage::ComputeStorage,
};
use wgpu::BufferUsages;

/// Size buckets for uniform buffer pooling.
/// Uniforms are typically 16-64 bytes; we round up to these sizes.
const UNIFORM_BUCKET_SIZES: [u64; 5] = [16, 32, 64, 128, 256];

/// A simple pool of uniform buffers to avoid per-dispatch wgpu::Device::create_buffer() calls.
/// Buffers are bucketed by size and recycled after each flush.
#[derive(Debug)]
struct UniformPool {
    device: wgpu::Device,
    /// Free buffers available for reuse, indexed by bucket (matching UNIFORM_BUCKET_SIZES).
    free: [Vec<wgpu::Buffer>; 5],
    /// Buffers currently in use (returned to free on release).
    in_use: Vec<(usize, wgpu::Buffer)>, // (bucket_index, buffer)
    /// Alignment requirement for uniform buffers.
    alignment: u64,
}

impl UniformPool {
    fn new(device: wgpu::Device, alignment: u64) -> Self {
        Self {
            device,
            free: [vec![], vec![], vec![], vec![], vec![]],
            in_use: Vec::new(),
            alignment,
        }
    }

    /// Find the bucket index for a given size, or None if too large.
    fn bucket_index(size: u64) -> Option<usize> {
        UNIFORM_BUCKET_SIZES.iter().position(|&s| s >= size)
    }

    /// Reserve a uniform buffer of at least `size` bytes.
    /// Returns a WgpuResource backed by a pooled (or newly created) buffer.
    fn reserve(&mut self, size: u64) -> WgpuResource {
        // Align size up to alignment requirement.
        let aligned_size = size.max(self.alignment).next_multiple_of(self.alignment);

        if let Some(idx) = Self::bucket_index(aligned_size) {
            let bucket_size = UNIFORM_BUCKET_SIZES[idx];
            let buffer = if let Some(buf) = self.free[idx].pop() {
                buf
            } else {
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Uniform Pool Buffer"),
                    size: bucket_size,
                    usage: BufferUsages::UNIFORM | BufferUsages::STORAGE | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            };
            let resource = WgpuResource::new(buffer.clone(), 0, size);
            self.in_use.push((idx, buffer));
            resource
        } else {
            // Size exceeds largest bucket — allocate without pooling.
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Uniform Oversize Buffer"),
                size: aligned_size,
                usage: BufferUsages::UNIFORM | BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let resource = WgpuResource::new(buffer, 0, size);
            // Not tracked for reuse — will be dropped.
            resource
        }
    }

    /// Return all in-use buffers to their respective free lists.
    fn release(&mut self) {
        for (idx, buffer) in self.in_use.drain(..) {
            self.free[idx].push(buffer);
        }
    }
}

#[derive(Debug)]
pub(crate) struct WgpuMemManager {
    memory_pool: MemoryManagement<WgpuStorage>,
    uniform_pool: UniformPool,
    memory_pool_staging: MemoryManagement<WgpuStorage>,
}

impl WgpuMemManager {
    pub(crate) fn new(
        device: wgpu::Device,
        memory_properties: MemoryDeviceProperties,
        memory_config: MemoryConfiguration,
        logger: Arc<ServerLogger>,
    ) -> Self {
        // Allocate storage & memory management for the main memory buffers. Any calls
        // to empty() or create() with a small enough size will be allocated from this
        // main memory pool.
        let memory_main = MemoryManagement::from_configuration(
            WgpuStorage::new(
                memory_properties.alignment as usize,
                device.clone(),
                BufferUsages::STORAGE
                    | BufferUsages::COPY_SRC
                    | BufferUsages::COPY_DST
                    | BufferUsages::INDIRECT,
            ),
            &memory_properties,
            memory_config,
            logger.clone(),
            MemoryManagementOptions::new("Main GPU Memory"),
        );

        let memory_staging = MemoryManagement::from_configuration(
            WgpuStorage::new(
                wgpu::COPY_BUFFER_ALIGNMENT as usize,
                device.clone(),
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            ),
            &memory_properties,
            // Unfortunately, we can't reuse a different part of a buffer for different reads, so we
            // can't have a single binding with multiple slices allocated.
            MemoryConfiguration::ExclusivePages,
            logger.clone(),
            MemoryManagementOptions::new("Staging CPU Memory").mode(MemoryAllocationMode::Auto),
        );

        let uniform_pool = UniformPool::new(
            device.clone(),
            memory_properties.alignment as u64,
        );

        Self {
            memory_pool: memory_main,
            memory_pool_staging: memory_staging,
            uniform_pool,
        }
    }

    pub(crate) fn reserve(&mut self, size: u64, stream_id: StreamId) -> Result<Handle, IoError> {
        Ok(Handle::new(
            self.memory_pool.reserve(size)?,
            None,
            None,
            stream_id,
            0,
            size,
        ))
    }

    pub(crate) fn reserve_staging(
        &mut self,
        size: u64,
    ) -> Result<(WgpuResource, SliceBinding), IoError> {
        let handle = self.memory_pool_staging.reserve(size)?;
        let binding = MemoryHandle::binding(handle);
        let resource = self
            .memory_pool_staging
            .get_resource(binding.clone(), None, None)
            .unwrap();

        Ok((resource, binding))
    }

    pub(crate) fn get_resource(&mut self, binding: Binding) -> Result<WgpuResource, IoError> {
        let handle = self
            .memory_pool
            .get(binding.memory.clone())
            .ok_or_else(|| IoError::InvalidHandle {
                backtrace: BackTrace::capture(),
            })?;
        let handle = match binding.offset_start {
            Some(offset) => handle.offset_start(offset),
            None => handle,
        };
        let handle = match binding.offset_end {
            Some(offset) => handle.offset_end(offset),
            None => handle,
        };
        Ok(self.memory_pool.storage().get(&handle))
    }

    pub(crate) fn reserve_uniform(&mut self, size: u64) -> WgpuResource {
        self.uniform_pool.reserve(size)
    }

    pub(crate) fn memory_usage(&self) -> cubecl_runtime::memory_management::MemoryUsage {
        self.memory_pool.memory_usage()
    }

    pub(crate) fn memory_cleanup(&mut self, explicit: bool) {
        self.memory_pool.cleanup(explicit);
    }

    pub(crate) fn mode(&mut self, mode: MemoryAllocationMode) {
        self.memory_pool.mode(mode);
    }

    pub(crate) fn release_uniforms(&mut self) {
        self.uniform_pool.release();
    }
}
