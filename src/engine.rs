use std::time::Instant;
use crate::renderer::Renderer;

pub struct Time {
    start_time: Instant,
    frame_time: Instant,
    elapsed_time: f32,
}

impl Time {
    pub fn new() -> Time {
        Time {
            start_time: Instant::now(),
            frame_time: Instant::now(),
            elapsed_time: 0.0,
        }
    }

    pub fn update(&mut self) {
        self.elapsed_time = self.frame_time.elapsed().as_secs_f32();
        self.frame_time = Instant::now();
    }

    pub fn time_since_start(&self) -> f32 {
        self.start_time.elapsed().as_secs_f32()
    }

    pub fn elapsed_time(&self) -> f32 { self.elapsed_time }
}

#[derive(Debug, Default)]
pub struct DebugStats {
    pub pixels_drawn: u32,
    pub triangles_input: u32,
    pub triangles_after_clipping: u32,
    pub triangles_clipping_extra: u32,
    pub rasterizer_input: u32,
    pub triangles_rasterized: u32,
    pub debug_values: Vec<(String, String)>,
    pub pixel_overdraw: u32,
    pub pixels_failed_z_test: u32,
    pub bsp_nodes_traversed: u32,
    pub leafs_visible: u32,
    pub faces_rendered: u32,
    pub models_rendered: u32,
}

impl DebugStats {
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    pub fn push_stat(&mut self, key: &str, value: &str) {
        self.debug_values.push((key.to_string(), value.to_string()));
    }
}

pub struct Engine {
    time: Time,
    renderer: Renderer,
}

impl Engine {
    pub fn update(&mut self) {
        self.time.update();
    }
}

impl Engine {
    pub fn renderer(&self) -> &Renderer {
        &self.renderer
    }
}

impl Engine {
    pub fn new(renderer: Renderer) -> Engine {
        Engine {
            time: Time::new(),
            renderer,
        }
    }

    pub fn time(&self) -> &Time {
        &self.time
    }
}