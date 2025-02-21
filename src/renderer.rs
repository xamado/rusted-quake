use glam::{IVec2, Mat4, Vec2, Vec3};
use crate::color::Color;
use crate::model::Model;

#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    position: Vec3,
    normal: Vec3,
    screen_position: Vec3,
}

impl Default for Vertex {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            normal: Vec3::ZERO,
            screen_position: Vec3::ZERO,
        }
    }
}

pub struct Renderer {
    screen_width: usize,
    screen_height: usize,
    back_buffer: Vec<u32>,
    depth_buffer: Vec<f32>,
}

impl Renderer {
    pub fn new(screen_width: usize, screen_height: usize) -> Self {
        let back_buffer: Vec<u32> = vec![0; screen_width * screen_height];
        let depth_buffer = vec![0.0; screen_width * screen_height];

        Self {
            screen_width,
            screen_height,
            back_buffer,
            depth_buffer,
        }
    }

    pub fn clear(&mut self) {
        self.back_buffer.fill(0);
        self.depth_buffer.fill(1.0);
    }

    pub fn get_back_buffer(&self) -> &[u32] {
        &self.back_buffer
    }

    pub fn draw(&mut self, vertex_buffer: &Vec<Vec3>, index_buffer: &Vec<u32>, w: &Mat4, wvp: &Mat4) {
        // draw indices as triangles, grab triples at a time
        for tri in index_buffer.chunks_exact(3) {
            let mut vertices: [Vertex; 3] = [Vertex::default(); 3];

            // temp: calculate normal for triangle
            let p0: Vec3 = vertex_buffer[tri[0] as usize];
            let p1: Vec3 = vertex_buffer[tri[1] as usize];
            let p2: Vec3 = vertex_buffer[tri[2] as usize];
            let normal = ((p1 - p0).cross(p2 - p0)).normalize();

            let mut visible = false;
            let mut clipped = false;

            for i in 0..3 {
                let obj_position: Vec3 = vertex_buffer[tri[i] as usize];
                let obj_normal: Vec3 = normal;

                let obj_position_4d = obj_position.extend(1.0);
                let clip = wvp.mul_vec4(obj_position_4d);

                if clip.w < 0.0 {
                    clipped = true;
                    continue;
                }

                // let ndc = wvp.project_point3(obj_position);

                // perspective divide to go from clip->NDC
                let ndc = clip.truncate() / clip.w;

                if !visible && ndc.x > -1.0 && ndc.x < 1.0 && ndc.y > -1.0 && ndc.y < 1.0 {
                    visible = true;
                }

                let w_position = w.transform_point3(obj_position);
                let w_normal = w.transform_vector3(obj_normal); // leaving this here since in theory this should be different per-vertex and come from the 3d mesh

                let p_screen = Vec3::new(
                    ((ndc.x + 1.0) * 0.5 * self.screen_width as f32),
                    ((1.0 - ndc.y) * 0.5 * self.screen_height as f32),
                    ndc.z
                );

                // println!("{:?}", p_screen);

                let vertex_input = Vertex {
                    screen_position: p_screen,
                    position: w_position,
                    normal: w_normal,
                };

                vertices[i] = vertex_input;
            }

            if !visible || clipped { // all 3 vertices are off-screen
                continue;
            }

            // self.rasterize_triangle(vertices[0], vertices[1], vertices[2]);
            self.rasterize_triangle(vertices[2], vertices[1], vertices[0]);
        }
    }

    fn pixel_shader(v: Vertex) -> u32 {
        let light_dir = Vec3::new(0.0, -1.0, -1.0);

        let intensity = v.normal.dot(light_dir);

        // let c: Color = Color::from_f32(1.0 * intensity, 1.0 * intensity, 1.0 * intensity, 1.0 * intensity);

        let c: Color = Color::from_f32(v.normal.x.abs(), v.normal.y.abs(), v.normal.z.abs(), 1.0);

        // return the color
        c.to_u32()
    }

    fn orient2d(a: Vec3, b: Vec3, c: Vec3) -> f32 {
        // (b.x - a.x) as i32 * (c.y - a.y as i32) - (b.y - a.y) as i32 * (c.x - a.x as i32)
        (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x)
    }

    fn rasterize_triangle(&mut self, v0: Vertex, v1: Vertex, v2: Vertex) {
        // compute bounding box
        let mut min_x: i32 = v0.screen_position.x.min(v1.screen_position.x).min(v2.screen_position.x) as i32;
        let mut min_y: i32 = v0.screen_position.y.min(v1.screen_position.y).min(v2.screen_position.y) as i32;
        let mut max_x: i32 = v0.screen_position.x.max(v1.screen_position.x).max(v2.screen_position.x) as i32;
        let mut max_y: i32 = v0.screen_position.y.max(v1.screen_position.y).max(v2.screen_position.y) as i32;

        // clip it against screen
        min_x = min_x.max(0);
        min_y = min_y.max(0);
        max_x = max_x.min(self.screen_width as i32 - 1);
        max_y = max_y.min(self.screen_height as i32 - 1);

        let area: f32 = Self::orient2d(v0.screen_position, v1.screen_position, v2.screen_position);

        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let p = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, 0.0);

                // determine barycentric coordinates for this pixel
                let w0 = Self::orient2d(v1.screen_position, v2.screen_position, p) / area;
                let w1 = Self::orient2d(v2.screen_position, v0.screen_position, p) / area;
                let w2 = Self::orient2d(v0.screen_position, v1.screen_position, p) / area;

                // let weight = w0 + w1 + w2;

                let interpolated_screen = (w0 * v0.screen_position + w1 * v1.screen_position + w2 * v2.screen_position);
                let interpolated_position = (w0 * v0.position + w1 * v1.position + w2 * v2.position);
                let interpolated_normal = (w0 * v0.normal + w1 * v1.normal + w2 * v2.normal);

                if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                    let pixel_index: usize = p.y as usize * self.screen_width + p.x as usize;

                    // now we need to calculate the interpolated vertex attributes... fun!
                    let v: Vertex = Vertex {
                        screen_position: interpolated_screen,
                        position: interpolated_position,
                        normal: interpolated_normal,
                    };

                    // let zbuffer_value = self.depth_buffer[pixel_index];
                    // if interpolated_position.z <= zbuffer_value {
                    //     continue;
                    // }

                    // run the "pixel shader" for this pixel
                    let c = Self::pixel_shader(v);

                    self.back_buffer[pixel_index] = c;
                    self.depth_buffer[pixel_index] = interpolated_position.z;
                }
            }
        }
    }
}