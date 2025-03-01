pub struct BackBuffer {
    width: u32,
    height: u32,
    back_buffer: Vec<u32>,
    depth_buffer: Vec<f32>,
}

impl BackBuffer {
    pub fn new(width: u32, height: u32) -> BackBuffer {
        let back_buffer: Vec<u32> = vec![0; (width * height) as usize];
        let depth_buffer = vec![0.0; (width * height) as usize];

        BackBuffer {
            width,
            height,
            back_buffer,
            depth_buffer
        }
    }
}

impl BackBuffer {
    pub fn get_back_buffer(&mut self) -> &mut [u32] {
        &mut self.back_buffer
    }

    pub fn get_depth_buffer(&mut self) -> &mut [f32] {
        &mut self.depth_buffer
    }

    pub fn get_width(&self) -> u32 { self.width }

    pub fn get_height(&self) -> u32 { self.height }
}