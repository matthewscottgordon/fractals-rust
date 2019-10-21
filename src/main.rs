use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::device::{Device, Features};
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass};
use vulkano::image::SwapchainImage;
use vulkano::instance::{Instance, PhysicalDevice};
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::swapchain;
use vulkano::swapchain::{AcquireError, SwapchainCreationError};
use vulkano::sync;
use vulkano::sync::{FlushError, GpuFuture};

use vulkano_win::VkSurfaceBuild;

use winit::{Event, EventsLoop, Window, WindowBuilder, WindowEvent};

use std::sync::Arc;

fn main() {
    let instance = {
        let extensions = vulkano_win::required_extensions();
        Instance::new(None, &extensions, None).expect("Failed to create Vulkan instance")
    };
    let physical_device = PhysicalDevice::enumerate(&instance)
        .next()
        .expect("No vulkan devices found");
    println!(
        "Using {} (Vulkan version {})",
        physical_device.name(),
        physical_device.api_version()
    );
    let queue_family = physical_device
        .queue_families()
        .find(|&q| q.supports_graphics())
        .expect("Couldn't find graphical queue family.");
    let (device, mut queues) = {
        let device_ext = vulkano::device::DeviceExtensions {
            khr_swapchain: true,
            ..vulkano::device::DeviceExtensions::none()
        };
        Device::new(
            physical_device,
            &Features::none(),
            &device_ext,
            [(queue_family, 0.5)].iter().cloned(),
        )
        .expect("Failed to create device.")
    };
    let queue = queues.next().unwrap();

    let mut events_loop = EventsLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&events_loop, instance.clone())
        .unwrap();
    let window = surface.window();

    let caps = surface
        .capabilities(physical_device)
        .expect("Failed to get surface capabilities");
    let dimensions = caps.current_extent.unwrap_or([1280, 1024]);
    let alpha = caps.supported_composite_alpha.iter().next().unwrap();
    let format = caps.supported_formats[0].0;

    use vulkano::swapchain::{PresentMode, SurfaceTransform, Swapchain};

    let (mut swapchain, images) = Swapchain::new(
        device.clone(),
        surface.clone(),
        caps.min_image_count,
        format,
        dimensions,
        1,
        caps.supported_usage_flags,
        &queue,
        SurfaceTransform::Identity,
        alpha,
        PresentMode::Fifo,
        true,
        None,
    )
    .expect("Failed to create swapchain");

    #[derive(Default, Debug, Clone)]
    struct Vertex {
        position: [f32; 2],
    }
    vulkano::impl_vertex!(Vertex, position);

    let vertex_buffer = {
        CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            [
                Vertex {
                    position: [-1.0, 1.0],
                },
                Vertex {
                    position: [1.0, 1.0],
                },
                Vertex {
                    position: [1.0, -1.0],
                },
                Vertex {
                    position: [1.0, -1.0],
                },
                Vertex {
                    position: [-1.0, -1.0],
                },
                Vertex {
                    position: [-1.0, 1.0],
                },
            ]
            .iter()
            .cloned(),
        )
        .unwrap()
    };

    mod vertex_shader {
        vulkano_shaders::shader! {
            ty: "vertex",
            src: "
#version 450

layout(location=0) in vec2 position;

void main() {
   gl_Position = vec4(position, 0.0, 1.0);
}"
        }
    }

    mod fragment_shader {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: "
#version 450

layout(location = 0) out vec4 colour;

vec2 mult_complex( vec2 a, vec2 b )
{
    return vec2( a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x );
}

vec3 julia_color( vec2 v, vec2 c)
{
    int i = 0;
    for(; i<1000 && length(v) < 2.0; ++i)
        v = mult_complex(v, v) + c;
    float i_scaled = min(1.0, float(i)/200.0);
    if(length(v) > 2.0)
        return vec3(i_scaled, i_scaled, i_scaled);
    return vec3(0.0, 0.0, 0.0);
}

vec2 complex_exp( vec2 a )
{
    return exp(a.x)*vec2(cos(a.y), sin(a.y));
}

void main() {
   float time = 1.0;
   vec2 viewport_dimensions = vec2(1000.0, 1000.0);
   vec2 phase = 0.7785 * complex_exp(vec2(0.0, 1.0)*time);
   colour = vec4(julia_color((gl_FragCoord.xy / viewport_dimensions) * 4.0 - 2.0, phase), 1.0);
}"
        }
    }

    let vertex_shader = vertex_shader::Shader::load(device.clone()).unwrap();
    let fragment_shader = fragment_shader::Shader::load(device.clone()).unwrap();

    let render_pass = Arc::new(
        vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: swapchain.format(),
                samples: 1,
            }
        },
        pass: {
            color: [color],
                depth_stencil: {}
        }
        )
        .unwrap(),
    );

    let pipeline = Arc::new(
        GraphicsPipeline::start()
            .vertex_input_single_buffer::<Vertex>()
            .vertex_shader(vertex_shader.main_entry_point(), ())
            .triangle_list()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(fragment_shader.main_entry_point(), ())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap(),
    );

    let mut dynamic_state = DynamicState {
        line_width: None,
        viewports: None,
        scissors: None,
    };

    let mut framebuffers =
        window_size_dependent_setup(&images, render_pass.clone(), &mut dynamic_state);

    let mut recreate_swapchain = false;

    let mut previous_frame_end = Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>;

    loop {
        previous_frame_end.cleanup_finished();

        if recreate_swapchain {
            let dimensions = if let Some(dimensions) = window.get_inner_size() {
                let dimensions: (u32, u32) =
                    dimensions.to_physical(window.get_hidpi_factor()).into();
                [dimensions.0, dimensions.1]
            } else {
                return;
            };

            let (new_swapchain, new_images) = match swapchain.recreate_with_dimension(dimensions) {
                Ok(r) => r,
                // This error tends to happen when the user is manually resizing the window.
                // Simply restarting the loop is the easiest way to fix this issue.
                Err(SwapchainCreationError::UnsupportedDimensions) => continue,
                Err(err) => panic!("{:?}", err),
            };

            swapchain = new_swapchain;
            // Because framebuffers contains an Arc on the old swapchain, we need to
            // recreate framebuffers as well.
            framebuffers =
                window_size_dependent_setup(&new_images, render_pass.clone(), &mut dynamic_state);

            recreate_swapchain = false;
        }

        let (image_num, acquire_future) =
            match swapchain::acquire_next_image(swapchain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    recreate_swapchain = true;
                    continue;
                }
                Err(err) => panic!("{:?}", err),
            };

        let clear_values = vec![[1.0, 0.0, 0.0, 1.0].into()];

        let command_buffer =
            AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family())
                .unwrap()
                .begin_render_pass(framebuffers[image_num].clone(), false, clear_values)
                .unwrap()
                .draw(
                    pipeline.clone(),
                    &dynamic_state,
                    vertex_buffer.clone(),
                    (),
                    (),
                )
                .unwrap()
                .end_render_pass()
                .unwrap()
                .build()
                .unwrap();

        let future = previous_frame_end
            .join(acquire_future)
            .then_execute(queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                previous_frame_end = Box::new(future) as Box<_>;
            }
            Err(FlushError::OutOfDate) => {
                recreate_swapchain = true;
                previous_frame_end = Box::new(sync::now(device.clone())) as Box<_>;
            }
            Err(e) => {
                println!("{:?}", e);
                previous_frame_end = Box::new(sync::now(device.clone())) as Box<_>;
            }
        }

        let mut done = false;
        events_loop.poll_events(|ev| match ev {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => done = true,
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => recreate_swapchain = true,
            _ => {}
        });
        if done {
            return;
        }
    }
}

fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    dynamic_state: &mut DynamicState,
) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
    let dimensions = images[0].dimensions();

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [dimensions[0] as f32, dimensions[1] as f32],
        depth_range: 0.0..1.0,
    };
    dynamic_state.viewports = Some(vec![viewport]);

    images
        .iter()
        .map(|image| {
            Arc::new(
                Framebuffer::start(render_pass.clone())
                    .add(image.clone())
                    .unwrap()
                    .build()
                    .unwrap(),
            ) as Arc<dyn FramebufferAbstract + Send + Sync>
        })
        .collect::<Vec<_>>()
}
