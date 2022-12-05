#![allow(clippy::all)]

use compute_engine::{BaseEngine, ComputeEngine};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::AutoCommandBufferBuilder,
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
};

mod shader {
    vulkano_shaders::shader! {ty: "compute", path: "src/shader.comp"}
}

#[cfg(test)]
mod tests;

pub fn entrypoint() {
    // Start Compute Engine
    let compute_engine = ComputeEngine::new();

    // Print information
    ComputeEngine::print_api_information(compute_engine.get_instance(), log::Level::Info);

    // Prepare Data
    let data_iter = 0..65536;
    let data_buffer = CpuAccessibleBuffer::from_iter(
        compute_engine.get_logical_device().get_device(),
        BufferUsage {
            storage_buffer: true,
            ..Default::default()
        },
        false,
        data_iter,
    )
    .expect("failed to create buffer");

    // Prepare Shader
    let shader = shader::load(compute_engine.get_logical_device().get_device())
        .expect("failed to create shader module");

    // Prepare Compute Pipeline
    let compute_pipeline = ComputePipeline::new(
        compute_engine.get_logical_device().get_device(),
        shader.entry_point("main").unwrap(),
        &(),
        None,
        |_| {},
    )
    .expect("failed to create compute pipeline");

    // Prepare Descriptor Set
    let layout = compute_pipeline.layout().set_layouts().get(0).unwrap();
    let set = PersistentDescriptorSet::new(
        layout.clone(),
        [WriteDescriptorSet::buffer(0, data_buffer.clone())],
    )
    .expect("failed to create descriptor set");

    // Submit Command Buffer for Computation
    compute_engine.compute(&|engine: &ComputeEngine| {
        let mut builder = AutoCommandBufferBuilder::primary(
            engine.get_logical_device().get_device(),
            engine.get_logical_device().get_queue_family_index(),
            vulkano::command_buffer::CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .bind_pipeline_compute(compute_pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                compute_pipeline.layout().clone(),
                0,
                set.clone(),
            )
            .dispatch([1024, 1, 1])
            .unwrap();

        builder.build().unwrap()
    });

    // Assert results
    let content = data_buffer.read().unwrap();
    for (n, val) in content.iter().enumerate() {
        assert_eq!(*val, n as u32 * 12);
    }
    log::info!("Assertion passed");
}
