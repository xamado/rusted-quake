use rand::{SeedableRng};
use crate::color::Color;
use glam::{vec3, IVec2, Mat4, Vec3, Vec4};
use rand::rngs::StdRng;

#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    position: Vec3,
    normal: Vec3,
    screen_position: Vec3,
}

pub struct VertexOutput {
    position: Vec4,
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
    pub debug_values: Vec<(String, String)>,
}

impl RendererStats {
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    pub fn push_stat(&mut self, key: &str, value: &str) {
        self.debug_values.push((key.to_string(), value.to_string()));
    }
}

pub struct RendererSettings {
    pub naive_rasterization: bool,
    pub tile_size: i32,
    pub wireframe: bool,
}

impl Default for RendererSettings {
    fn default() -> Self {
        Self {
            naive_rasterization: false,
            tile_size: 16,
            wireframe: false,
        }
    }
}

pub struct Renderer {
    screen_width: usize,
    screen_height: usize,
    back_buffer: Vec<u32>,
    depth_buffer: Vec<f32>,
    stats: RendererStats,
    settings: RendererSettings,
    rng: StdRng,
}

impl Renderer {
    pub fn new(screen_width: usize, screen_height: usize, settings: RendererSettings) -> Self {
        let back_buffer: Vec<u32> = vec![0; screen_width * screen_height];
        let depth_buffer = vec![0.0; screen_width * screen_height];

        let rng = StdRng::seed_from_u64(12345);

        Self {
            screen_width,
            screen_height,
            back_buffer,
            depth_buffer,
            stats: RendererStats::default(),
            settings,
            rng,
        }
    }

    pub fn clear(&mut self) {
        self.back_buffer.fill(0);
        self.depth_buffer.fill(1.0);

        self.stats.clear();

        self.rng = StdRng::seed_from_u64(12345);
    }

    pub fn get_stats(&self) -> &RendererStats {
        &self.stats
    }

    pub fn get_settings(&self) -> &RendererSettings { &self.settings }

    pub fn get_back_buffer(&self) -> &[u32] {
        &self.back_buffer
    }

    pub fn get_depth_buffer(&self) -> &[f32] {
        &self.depth_buffer
    }

    // The Vertex Shader Stage is responsible for transforming the object space vertices
    // all the way into clip-space.
    // Perspective divide and Clipping are not performed at this stage on a GPU, but rather
    // at the fixed-function part of the pipeline after the VS stage.
    fn vertex_stage(&self, vertices: &Vec<Vertex>, wvp: &Mat4, world: &Mat4) -> Vec<VertexOutput> {
        let mut output: Vec<VertexOutput> = vec![];

        for vertex in vertices.iter() {
            // Object space -> Clip space
            let clip_position = wvp.mul_vec4(vertex.position.extend(1.0));

            // Object space -> World space
            let w_normal = world.transform_vector3(vertex.normal);

            output.push(VertexOutput {
                position: clip_position,
                normal: w_normal,
                screen_position: Vec3::ZERO,
            });
        }

        output
    }

    pub fn draw(&mut self, vertex_buffer_positions: &Vec<Vec3>, index_buffer: &Vec<u32>, w: &Mat4, wvp: &Mat4) { 
        // TEMP: Generate our Vertex vertex_buffer here
        let mut vertex_buffer: Vec<Vertex> = vec![Vertex::default(); vertex_buffer_positions.len()];

        // TEMP: Generate normals for our vertices
        for tri in index_buffer.chunks_exact(3) {
            let i0 = tri[0] as usize;
            let i1 = tri[1] as usize;
            let i2 = tri[2] as usize;

            let p0: Vec3 = vertex_buffer_positions[i0];
            let p1: Vec3 = vertex_buffer_positions[i1];
            let p2: Vec3 = vertex_buffer_positions[i2];

            let normal = (p1 - p0).cross(p2 - p0).normalize();

            vertex_buffer[i0].position = p0;
            vertex_buffer[i0].normal = normal;
            vertex_buffer[i1].position = p1;
            vertex_buffer[i1].normal = normal;
            vertex_buffer[i2].position = p2;
            vertex_buffer[i2].normal = normal;
        }

        // Vertex Shader Stage -> Transform our vertices to clip space
        let vertex_buffer_clip: Vec<VertexOutput> = self.vertex_stage(&vertex_buffer, &wvp, &w);

        // draw indices as triangles, grab triples at a time
        for tri in index_buffer.chunks_exact(3) {
            self.stats.triangles_input += 1;

            let v0 = &vertex_buffer_clip[tri[0] as usize];
            let v1 = &vertex_buffer_clip[tri[1] as usize];
            let v2 = &vertex_buffer_clip[tri[2] as usize];

            // clip when the vertex falls behind the near plane
            if v0.position.w + v0.position.z < 0.0 || v1.position.w + v1.position.z < 0.0 || v2.position.w + v2.position.z < 0.0 {
                continue; // clipped
            }

            let v0_ndc = v0.position / v0.position.w;
            let v1_ndc = v1.position / v1.position.w;
            let v2_ndc = v2.position / v2.position.w;

            // ignore this clipping, we will rasterize anyway, just use guard bands to prevent overflow
            // if ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 {
            //     clipped |= 1 << i;
            // }

            // viewport transformation
            let v0_screen = vec3(
                (v0_ndc.x + 1.0) * 0.5 * self.screen_width as f32,
                (1.0 - v0_ndc.y) * 0.5 * self.screen_height as f32,
                v0_ndc.z
            );

            let v1_screen = vec3(
                (v1_ndc.x + 1.0) * 0.5 * self.screen_width as f32,
                (1.0 - v1_ndc.y) * 0.5 * self.screen_height as f32,
                v1_ndc.z
            );

            let v2_screen = vec3(
                (v2_ndc.x + 1.0) * 0.5 * self.screen_width as f32,
                (1.0 - v2_ndc.y) * 0.5 * self.screen_height as f32,
                v2_ndc.z
            );

            // guard-band clipping
            let guardband = 8192.0 * 4.0;
            if v0_screen.x < -guardband || v0_screen.x > guardband || v0_screen.y < -guardband || v0_screen.y > guardband {
                continue; // clipped
            }
            if v1_screen.x < -guardband || v1_screen.x > guardband || v1_screen.y < -guardband || v1_screen.y > guardband {
                continue; // clipped
            }
            if v2_screen.x < -guardband || v2_screen.x > guardband || v2_screen.y < -guardband || v2_screen.y > guardband {
                continue; // clipped
            }

            let v0_rasterizer = VertexOutput {
                position: v0.position,
                normal: v0.normal,
                screen_position: v0_screen,
            };

            let v1_rasterizer = VertexOutput {
                position: v1.position,
                normal: v1.normal,
                screen_position: v1_screen,
            };

            let v2_rasterizer = VertexOutput {
                position: v2.position,
                normal: v2.normal,
                screen_position: v2_screen,
            };

            // Send to the rasterizer
            if self.settings.naive_rasterization {
                self.rasterize_triangle_naive(&v2_rasterizer, &v1_rasterizer, &v0_rasterizer);
            }
            else {
                self.rasterize_triangle(&v2_rasterizer, &v1_rasterizer, &v0_rasterizer);
            }
        }
    }

    fn pixel_shader(v: VertexOutput) -> u32 {
        // let light_dir = Vec3::new(0.0, -1.0, -1.0);
        // let intensity = v.normal.dot(light_dir);
        // let c: Color = Color::from_f32(1.0 * intensity, 1.0 * intensity, 1.0 * intensity, 1.0 * intensity);

        let c: Color = Color::from_f32(v.normal.x.abs(), v.normal.y.abs(), v.normal.z.abs(), 1.0);

        // return the color
        c.to_u32()
    }

    fn edge_function(v0: IVec2, v1: IVec2, p: IVec2) -> i32 {
        (v1.x - v0.x) * (p.y - v0.y) - (v1.y - v0.y) * (p.x - v0.x)
    }

    fn rasterize_triangle(&mut self, v0: &VertexOutput, v1: &VertexOutput, v2: &VertexOutput) {
        self.stats.rasterizer_input += 1;
        let tile_size = self.settings.tile_size;

        // move screen position of our vertices to integer coordinates
        let iv0 = IVec2::new(v0.screen_position.x.ceil() as i32, v0.screen_position.y.ceil() as i32);
        let iv1 = IVec2::new(v1.screen_position.x.ceil() as i32, v1.screen_position.y.ceil() as i32);
        let iv2 = IVec2::new(v2.screen_position.x.ceil() as i32, v2.screen_position.y.ceil() as i32);

        // calculate the area of the triangle (2x)
        let area: i32 = Self::edge_function(iv0, iv1, iv2);
        if area <= 0 { // negative area means it's a back-facing triangle
            return;
        }

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


        // pre-calculate the area for the first pixel in the grid
        let p_min = IVec2::new(min_x, min_y);
        let w0_min = Self::edge_function(iv1, iv2, p_min);
        let w1_min = Self::edge_function(iv2, iv0, p_min);
        let w2_min = Self::edge_function(iv0, iv1, p_min);

        // and calculate steps needed to move the barycentric coordinates
        let w0_stepx = iv1.y - iv2.y;
        let w0_stepy = iv2.x - iv1.x;
        let w1_stepx = iv2.y - iv0.y;
        let w1_stepy = iv0.x - iv2.x;
        let w2_stepx = iv0.y - iv1.y;
        let w2_stepy = iv1.x - iv0.x;

        for tile_y in (min_y..=max_y).step_by(tile_size as usize) {
            for tile_x in (min_x..=max_x).step_by(tile_size as usize) {
                let tile_row = tile_y - min_y;
                let tile_col = tile_x - min_x;

                let tile_end_x = (tile_x + tile_size).min(max_x);
                let tile_end_y = (tile_y + tile_size).min(max_y);

                let mut tile_w0_min = w0_min + w0_stepy * tile_row + w0_stepx * tile_col;
                let mut tile_w1_min = w1_min + w1_stepy * tile_row + w1_stepx * tile_col;
                let mut tile_w2_min = w2_min + w2_stepy * tile_row + w2_stepx * tile_col;

                let tile_corners = [
                    IVec2::new(tile_x, tile_y),  // Top-left
                    IVec2::new(tile_end_x, tile_y),  // Top-right
                    IVec2::new(tile_x, tile_end_y),  // Bottom-left
                    IVec2::new(tile_end_x, tile_end_y),  // Bottom-right
                ];

                let mut inside_count = 0;
                for &corner in tile_corners.iter() {
                    let w0 = w0_min + w0_stepy * (corner.y - min_y) + w0_stepx * (corner.x - min_x);
                    let w1 = w1_min + w1_stepy * (corner.y - min_y) + w1_stepx * (corner.x - min_x);
                    let w2 = w2_min + w2_stepy * (corner.y - min_y) + w2_stepx * (corner.x - min_x);

                    if (w0 | w1 | w2) >= 0 {
                        inside_count += 1;
                    }
                }

                // TODO skip tile if we prove it's completely outside...
                // if inside_count == 0 {
                //     continue;
                // }

                if inside_count == 4 { // if it's fully inside a triangle
                    // since this tile is completely inside the triangle, we can linearly
                    // interpolate vertex attributes, making it cheaper.
                    let tile_l0_tl = tile_w0_min as f32 / area as f32;
                    let tile_l1_tl = tile_w1_min as f32 / area as f32;
                    let tile_l2_tl = tile_w2_min as f32 / area as f32;

                    let tile_l0_tr = (tile_w0_min + w0_stepx * tile_size) as f32 / area as f32;
                    let tile_l1_tr = (tile_w1_min + w1_stepx * tile_size) as f32 / area as f32;
                    let tile_l2_tr = (tile_w2_min + w2_stepx * tile_size) as f32 / area as f32;

                    let tile_l0_bl = (tile_w0_min + w0_stepy * tile_size) as f32 / area as f32;
                    let tile_l1_bl = (tile_w1_min + w1_stepy * tile_size) as f32 / area as f32;
                    let tile_l2_bl = (tile_w2_min + w2_stepy * tile_size) as f32 / area as f32;

                    let tile_position_tl = tile_l0_tl * v0.position + tile_l1_tl * v1.position + tile_l2_tl * v2.position;
                    let tile_position_tr = tile_l0_tr * v0.position + tile_l1_tr * v1.position + tile_l2_tr * v2.position;
                    let tile_position_bl = tile_l0_bl * v0.position + tile_l1_bl * v1.position + tile_l2_bl * v2.position;

                    let tile_normal_tl = tile_l0_tl * v0.normal + tile_l1_tl * v1.normal + tile_l2_tl * v2.normal;
                    let tile_normal_tr = tile_l0_tr * v0.normal + tile_l1_tr * v1.normal + tile_l2_tr * v2.normal;
                    let tile_normal_bl = tile_l0_bl * v0.normal + tile_l1_bl * v1.normal + tile_l2_bl * v2.normal;

                    let tile_position_step_x = (tile_position_tr - tile_position_tl) / tile_size as f32;
                    let tile_position_step_y = (tile_position_bl - tile_position_tl) / tile_size as f32;

                    let tile_normal_step_x = (tile_normal_tr - tile_normal_tl) / tile_size as f32;
                    let tile_normal_step_y = (tile_normal_bl - tile_normal_tl) / tile_size as f32;

                    let mut interpolated_position_min = tile_position_tl;
                    let mut interpolated_normal_min = tile_normal_tl;

                    for y in tile_y..=tile_end_y {
                        let mut w0 = tile_w0_min;
                        let mut w1 = tile_w1_min;
                        let mut w2 = tile_w2_min;

                        let mut interpolated_position = interpolated_position_min;
                        let mut interpolated_normal = interpolated_normal_min;

                        for x in tile_x..=tile_end_x {
                            let p = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, 0.0);

                            let pixel_index: usize = y as usize * self.screen_width + x as usize;

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


                            // run the "pixel shader" for this pixel
                            let c = Self::pixel_shader(VertexOutput {
                                screen_position: interpolated_screen,
                                position: interpolated_position,
                                normal: interpolated_normal,
                            });

                            self.back_buffer[pixel_index] = c;
                            self.depth_buffer[pixel_index] = pixel_z;

                            // inc stats
                            self.stats.pixels_drawn += 1;

                            // step our weights and attributes horizontally
                            w0 += w0_stepx;
                            w1 += w1_stepx;
                            w2 += w2_stepx;

                            interpolated_position += tile_position_step_x;
                            interpolated_normal += tile_normal_step_x;
                        }

                        // step our weights and attributes vertically
                        tile_w0_min += w0_stepy;
                        tile_w1_min += w1_stepy;
                        tile_w2_min += w2_stepy;

                        interpolated_position_min += tile_position_step_y;
                        interpolated_normal_min += tile_normal_step_y;
                    }
                }
                else { // it's partially inside/outside
                    for y in tile_y..=tile_end_y {
                        let mut w0 = tile_w0_min;
                        let mut w1 = tile_w1_min;
                        let mut w2 = tile_w2_min;

                        for x in tile_x..=tile_end_x {
                            let p = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, 0.0);

                            // let w0 = Self::edge_function(iv1, iv2, IVec2::new(x, y));
                            // let w1 = Self::edge_function(iv2, iv0, IVec2::new(x, y));
                            // let w2 = Self::edge_function(iv0, iv1, IVec2::new(x, y));

                            // skip if outside the triangle
                            if (w0 | w1 | w2) >= 0 {
                                let pixel_index: usize = y as usize * self.screen_width + x as usize;

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

                                let v = VertexOutput {
                                    screen_position: interpolated_screen,
                                    position: interpolated_position,
                                    normal: interpolated_normal,
                                };

                                // inc stats
                                self.stats.pixels_drawn += 1;

                                // run the "pixel shader" for this pixel
                                let c = Self::pixel_shader(v);

                                // let c = Color::from_u8(255,0,255, 255).to_u32();

                                self.back_buffer[pixel_index] = c;
                                self.depth_buffer[pixel_index] = pixel_z;
                            }

                            w0 += w0_stepx;
                            w1 += w1_stepx;
                            w2 += w2_stepx;
                        }

                        tile_w0_min += w0_stepy;
                        tile_w1_min += w1_stepy;
                        tile_w2_min += w2_stepy;
                    }
                }
            }
        }

        /*
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
        }*/
    }

    fn rasterize_triangle_naive(&mut self, v0: &VertexOutput, v1: &VertexOutput, v2: &VertexOutput) {
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
        let area: i32 = Self::edge_function(iv0, iv1, iv2);
        if area <= 0 { // negative area means it's a back-facing triangle
            return;
        }

        // pre-calculate the area for the first pixel in the grid
        let p_min = IVec2::new(min_x, min_y);
        let mut w0_min = Self::edge_function(iv1, iv2, p_min);
        let mut w1_min = Self::edge_function(iv2, iv0, p_min);
        let mut w2_min = Self::edge_function(iv0, iv1, p_min);

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

                    let v = VertexOutput {
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