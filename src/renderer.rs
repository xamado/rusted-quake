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

#[derive(Debug, Default)]
pub struct RendererStats {
    pixels_drawn: u32,
    triangles_input: u32,
    rasterizer_input: u32,
}

impl RendererStats {
    pub fn clear(&mut self) {
        *self = Self::default();
    }
}

pub struct Renderer {
    screen_width: usize,
    screen_height: usize,
    back_buffer: Vec<u32>,
    depth_buffer: Vec<f32>,
    stats: RendererStats,
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
            stats: RendererStats::default()
        }
    }

    pub fn clear(&mut self) {
        self.back_buffer.fill(0);
        self.depth_buffer.fill(1.0);

        self.stats.clear();
    }

    pub fn get_stats(&self) -> &RendererStats {
        &self.stats
    }

    pub fn get_back_buffer(&self) -> &[u32] {
        &self.back_buffer
    }

    pub fn get_depth_buffer(&self) -> &[f32] {
        &self.depth_buffer
    }

    pub fn draw(&mut self, vertex_buffer: &Vec<Vec3>, index_buffer: &Vec<u32>, w: &Mat4, wvp: &Mat4) {
        // draw indices as triangles, grab triples at a time
        for tri in index_buffer.chunks_exact(3) {
            self.stats.triangles_input += 1;

            let mut vertices: [Vertex; 3] = [Vertex::default(); 3];

            // temp: calculate normal for triangle
            let p0: Vec3 = vertex_buffer[tri[0] as usize];
            let p1: Vec3 = vertex_buffer[tri[1] as usize];
            let p2: Vec3 = vertex_buffer[tri[2] as usize];
            let normal = (p1 - p0).cross(p2 - p0).normalize();

            let mut visible = false;
            let mut clipped = false;

            for i in 0..3 {
                let obj_position: Vec3 = vertex_buffer[tri[i] as usize];
                let obj_normal: Vec3 = normal;

                // clip space transformation
                let obj_position_4d = obj_position.extend(1.0);
                let clip = wvp.mul_vec4(obj_position_4d);

                if clip.w < 0.0 {
                    clipped = true;
                    continue;
                }

                // NDC transformation -> perspective divide to go from clip->NDC
                let ndc = clip.truncate() / clip.w;

                if !visible && ndc.x > -1.0 && ndc.x < 1.0 && ndc.y > -1.0 && ndc.y < 1.0 {
                    visible = true;
                }

                let w_position = w.transform_point3(obj_position);
                let w_normal = w.transform_vector3(obj_normal); // leaving this here since in theory this should be different per-vertex and come from the 3d mesh

                // viewport transformation
                let p_screen = Vec3::new(
                    (ndc.x + 1.0) * 0.5 * self.screen_width as f32,
                    (1.0 - ndc.y) * 0.5 * self.screen_height as f32,
                    ndc.z
                );

                // println!("screen: {:?}", p_screen);

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
            self.rasterize_triangle(&vertices[2], &vertices[1], &vertices[0]);
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

    fn orient2d(v0: IVec2, v1: IVec2, p: IVec2) -> i32 {
        // (b.x - a.x) as i32 * (c.y - a.y as i32) - (b.y - a.y) as i32 * (c.x - a.x as i32)
        // (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x)

        (v1.x - v0.x) * (p.y - v0.y) - (v1.y - v0.y) * (p.x - v0.x) // this is how https://fgiesen.wordpress.com/2013/02/06/the-barycentric-conspirac/ does it
    }

    fn rasterize_triangle(&mut self, v0: &Vertex, v1: &Vertex, v2: &Vertex) {
        self.stats.rasterizer_input += 1;
        
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

        let iv0 = IVec2::new(v0.screen_position.x.ceil() as i32, v0.screen_position.y.ceil() as i32);
        let iv1 = IVec2::new(v1.screen_position.x.ceil() as i32, v1.screen_position.y.ceil() as i32);
        let iv2 = IVec2::new(v2.screen_position.x.ceil() as i32, v2.screen_position.y.ceil() as i32);

        // calculate the area of the triangle (2x)
        let area: i32 = Self::orient2d(iv0, iv1, iv2);
        if area < 0 { // negative area means it's a back-facing triangle
            return;
        }

        // pre-calculate the barycentric for the first pixel in the grid
        let p_min = IVec2::new(min_x, min_y);
        let mut w0_min = Self::orient2d(iv1, iv2, p_min);
        let mut w1_min = Self::orient2d(iv2, iv0, p_min);
        let mut w2_min = Self::orient2d(iv0, iv1, p_min);

        // and calculate steps needed to move the barycentric coordinates
        let w0_stepx = iv1.y - iv2.y;
        let w0_stepy = iv2.x - iv1.x;
        let w1_stepx = iv2.y - iv0.y;
        let w1_stepy = iv0.x - iv2.x;
        let w2_stepx = iv0.y - iv1.y;
        let w2_stepy = iv1.x - iv0.x;

        for y in min_y..=max_y {
            // initialize barycentric coords
            let mut w0 = w0_min;
            let mut w1 = w1_min;
            let mut w2 = w2_min;

            for x in min_x..=max_x {
                let p = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, 0.0);

                // skip if outside the triangle
                if (w0 | w1 | w2) >= 0 {
                    let pixel_index: usize = p.y as usize * self.screen_width + p.x as usize;

                    // normalize our lambdas and move to floating point math
                    let l0 = w0 as f32 / area as f32;
                    let l1 = w1 as f32 / area as f32;
                    let l2 = w2 as f32 / area as f32;

                    // calculate perspective-correct depth
                    let inv_z = (1.0 / v0.screen_position.z) * l0 + (1.0 / v1.screen_position.z) * l1 + (1.0 / v2.screen_position.z) * l2;
                    let pixel_z = 1.0 / inv_z;

                    // test against our z-buffer
                    let zbuffer_value = self.depth_buffer[pixel_index];
                    if pixel_z > zbuffer_value {
                        continue;
                    }

                    // now we need to calculate the interpolated vertex attributes... fun!
                    let interpolated_screen: Vec3 = Vec3::new(p.x, p.y, pixel_z);
                    let interpolated_position = l0 * v0.position + l1 * v1.position + l2 * v2.position;
                    let interpolated_normal = l0 * v0.normal + l1 * v1.normal + l2 * v2.normal;

                    let v: Vertex = Vertex {
                        screen_position: interpolated_screen,
                        position: interpolated_position,
                        normal: interpolated_normal,
                    };

                    // inc stats
                    self.stats.pixels_drawn += 1;

                    // run the "pixel shader" for this pixel
                    let c = Self::pixel_shader(v);

                    self.back_buffer[pixel_index] = c;
                    self.depth_buffer[pixel_index] = pixel_z;
                }

                // take a step to the right
                w0 += w0_stepx;
                w1 += w1_stepx;
                w2 += w2_stepx;
            }

            // and a step down
            w0_min += w0_stepy;
            w1_min += w1_stepy;
            w2_min += w2_stepy;
        }
    }
}