use glam::{ivec2, vec3, vec4, IVec2, Mat4, Vec2, Vec3, Vec4, Vec4Swizzles};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::sync::Arc;
use crate::color::Color;

#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub color: Vec4,
    pub tex_coord: Vec2,
    pub uv_lightmap: Vec2,
}

pub struct Texture {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub data: Vec<Vec<u8>>,
    pub palette: Arc<[u32; 256]>,
}

#[derive(Copy, Clone, Debug)]
pub struct VertexOutput {
    position: Vec4,
    normal: Vec3,
    color: Vec4,
    texcoord: Vec2,
    uv_lightmap: Vec2,
    screen_position: Vec4,
    world_position: Vec3,
}

impl Default for Vertex {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            normal: Vec3::ZERO,
            color: Vec4::ZERO,
            tex_coord: Vec2::ZERO,
            uv_lightmap: Vec2::ZERO,
        }
    }
}

#[derive(Debug, Default)]
pub struct RendererStats {
    pixels_drawn: u32,
    triangles_input: u32,
    triangles_after_clipping: u32,
    triangles_clipping_extra: u32,
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
            let w_position = world.transform_point3(vertex.position);
            let w_normal = world.transform_vector3(vertex.normal);

            output.push(VertexOutput {
                position: clip_position,
                normal: w_normal,
                color: vertex.color,
                texcoord: vertex.tex_coord,
                uv_lightmap: vertex.uv_lightmap,
                screen_position: Vec4::ZERO,
                world_position: w_position,
            });
        }

        output
    }

    fn clip_edge(v1: &VertexOutput, v2: &VertexOutput, plane: &Vec4) -> VertexOutput {
        // let t = (- (v1.position.w + v1.position.z)) / ((v2.position.w + v2.position.z) - (v1.position.w + v1.position.z));

        let d_in = v1.position.dot(*plane);
        let d_out = v2.position.dot(*plane);
        let t = d_in / (d_in - d_out);

        let clipped_vertex = VertexOutput {
            position: v1.position + t * (v2.position - v1.position),
            normal: v1.normal + t * (v2.normal - v1.normal),
            color: v1.color + t * (v2.color - v1.color),
            texcoord: v1.texcoord + t * (v2.texcoord - v1.texcoord),
            uv_lightmap: v1.uv_lightmap + t * (v2.uv_lightmap - v1.uv_lightmap),
            screen_position: v1.screen_position + t * (v2.screen_position - v1.screen_position),
            world_position: v1.world_position + t * (v2.world_position - v1.world_position),
        };

        clipped_vertex
    }

    fn clip_triangle(v0: &VertexOutput, v1: &VertexOutput, v2: &VertexOutput) -> Vec<[VertexOutput;3]> {
        // Discard the triangle completely if off-screen. Our clipping routine won't handle this since we use expanded
        // guard-band planes to avoid clipping too early
        let outside_left   = v0.position.x < -v0.position.w && v1.position.x < -v1.position.w && v2.position.x < -v2.position.w;
        let outside_right  = v0.position.x >  v0.position.w && v1.position.x >  v1.position.w && v2.position.x >  v2.position.w;
        let outside_bottom = v0.position.y < -v0.position.w && v1.position.y < -v1.position.w && v2.position.y < -v2.position.w;
        let outside_top    = v0.position.y >  v0.position.w && v1.position.y >  v1.position.w && v2.position.y >  v2.position.w;
        let outside_near   = v0.position.z < -v0.position.w && v1.position.z < -v1.position.w && v2.position.z < -v2.position.w;
        let outside_far    = v0.position.z >  v0.position.w && v1.position.z >  v1.position.w && v2.position.z >  v2.position.w;
        if outside_left || outside_right || outside_bottom || outside_top || outside_near || outside_far {
            return vec![];
        }

        let mut triangles: Vec<[VertexOutput;3]> = vec![];
        triangles.push([*v0, *v1, *v2]);

        let guard_band_scale = 15.0;

        let clip_planes = [
            vec4(0.0, 0.0, 1.0, 1.0),  // Near Plane: z + w = 0
            vec4(1.0, 0.0, 0.0, guard_band_scale),  // Left Plane: -x + w = 0
            vec4(-1.0, 0.0, 0.0,guard_band_scale), // Right Plane: x + w = 0
            vec4(0.0, 1.0, 0.0, guard_band_scale),  // Bottom Plane: -y + w = 0
            vec4(0.0, -1.0, 0.0,guard_band_scale), // Top Plane: y + w = 0
        ];

        for plane in clip_planes {
            let mut new_triangles: Vec<[VertexOutput;3]> = vec![];

            for t in triangles.iter() {
                let clipped_triangles = Self::clip_triangle_plane(&t[0], &t[1], &t[2], &plane);
                new_triangles.extend_from_slice(&clipped_triangles);
            }

            triangles = new_triangles;
        }

        triangles
    }

    fn clip_triangle_plane(v0: &VertexOutput, v1: &VertexOutput, v2: &VertexOutput, plane: &Vec4) -> Vec<[VertexOutput; 3]> {
        let d0 = v0.position.dot(*plane);
        let d1 = v1.position.dot(*plane);
        let d2 = v2.position.dot(*plane);

        match ((d0 >= 0.0) as u8, (d1 >= 0.0) as u8, (d2 >= 0.0) as u8) {
            (1, 1, 1) => { // fully inside, nothing to clip
                vec![[*v0, *v1, *v2]]
            }
            (0, 1, 1) => { // v0 is outside
                let new0 = Self::clip_edge(v0, v1, plane);
                let new1 = Self::clip_edge(v0, v2, plane);
                vec![
                    [new0, *v1, *v2],
                    [new0, *v2, new1]
                ]
            }
            (1, 0, 1) => { // v1 is outside, replace it
                let new0 = Self::clip_edge(v1, v0, plane);
                let new1 = Self::clip_edge(v1, v2, plane);
                vec![
                    [*v0, new0, *v2],
                    [new0, new1, *v2]
                ]
            }
            (1, 1, 0) => { // v2 is outside
                let new0 = Self::clip_edge(v2, v0, plane);
                let new1 = Self::clip_edge(v2, v1, plane);
                vec![
                    [*v0, *v1, new0],
                    [new0, *v1, new1]
                ]
            }
            (0, 0, 1) => { // v2 is inside, v0 and v1 are outside
                let nv0 = Self::clip_edge(v2, v0, plane);
                let nv1 = Self::clip_edge(v2, v1, plane);
                vec![
                    [nv0, nv1, *v2]
                ]
            }
            (1, 0, 0) => { // v0 is inside, v1 and v2 are outside
                let nv1 = Self::clip_edge(v0, v1, plane);
                let nv2 = Self::clip_edge(v0, v2, plane);
                vec![
                    [*v0, nv1, nv2]
                ]
            }
            (0, 1, 0) => { // v0 and v2 are outside, v1 is inside
                let nv0 = Self::clip_edge(v1, v0, plane);
                let nv2 = Self::clip_edge(v1, v2, plane);
                vec![
                    [nv0, *v1, nv2]
                ]
            }
            (0, 0, 0) => { // Fully outside, discard triangle
                vec![]
            }
            _ => unreachable!(),
        }
    }

    pub fn draw(
        &mut self,
        vertex_buffer: &Vec<Vertex>,
        index_buffer: &Vec<u32>,
        w: &Mat4,
        wvp: &Mat4,
        texture: &Texture,
        lightmap: &[u8],
        lightmap_size: &IVec2,
    ) {
        // Vertex Shader Stage -> Transform our vertices to clip space
        let vertex_buffer_clip: Vec<VertexOutput> = self.vertex_stage(&vertex_buffer, &wvp, &w);

        // draw indices as triangles, grab triples at a time
        for tri in index_buffer.chunks_exact(3) {
            self.stats.triangles_input += 1;

            let clip_v0 = &vertex_buffer_clip[tri[0] as usize];
            let clip_v1 = &vertex_buffer_clip[tri[1] as usize];
            let clip_v2 = &vertex_buffer_clip[tri[2] as usize];

            // clip input triangles, output could be 1 or 2 triangles
            let triangles = Self::clip_triangle(clip_v0, clip_v1, clip_v2);

            self.stats.triangles_after_clipping += triangles.len() as u32;
            self.stats.triangles_clipping_extra += triangles.len().saturating_sub(1) as u32;

            for triangle_verts in triangles {
                // let normal = (triangle_verts[1].world_position - triangle_verts[0].world_position)
                //     .cross(triangle_verts[2].world_position - triangle_verts[0].world_position)
                //     .normalize();
                //
                // let ndc: Vec<Vec4> = triangle_verts.iter()
                //     .map(|v| v.position / v.position.w)
                //     .collect();
                //
                // let screen: Vec<Vec3> =

                let v0 = &triangle_verts[0];
                let v1 = &triangle_verts[1];
                let v2 = &triangle_verts[2];

                let normal = (v1.world_position - v0.world_position).cross(v2.world_position - v0.world_position).normalize();

                // clip when the vertex falls behind the near plane
                if v0.position.w + v0.position.z < 0.0 || v1.position.w + v1.position.z < 0.0 || v2.position.w + v2.position.z < 0.0 {
                    // self.stats.push_stat("clipped by strange case: ", "true");
                    println!("clipped by strange case: v0 {:?} | v1 {:?} | v2 {:?}", v0, v1, v2);
                    // unreachable!();
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
                let v0_screen = vec4(
                    (v0_ndc.x + 1.0) * 0.5 * self.screen_width as f32,
                    (1.0 - v0_ndc.y) * 0.5 * self.screen_height as f32,
                    v0_ndc.z,
                    v0.position.w,
                );

                let v1_screen = vec4(
                    (v1_ndc.x + 1.0) * 0.5 * self.screen_width as f32,
                    (1.0 - v1_ndc.y) * 0.5 * self.screen_height as f32,
                    v1_ndc.z,
                    v1.position.w,
                );

                let v2_screen = vec4(
                    (v2_ndc.x + 1.0) * 0.5 * self.screen_width as f32,
                    (1.0 - v2_ndc.y) * 0.5 * self.screen_height as f32,
                    v2_ndc.z,
                    v2.position.w,
                );

                // guard-band clipping
                let guardband = 8192.0 * 4.0;
                if v0_screen.x < -guardband || v0_screen.x > guardband || v0_screen.y < -guardband || v0_screen.y > guardband {
                    self.stats.push_stat("clipped by guardband", "v0");
                    continue; // clipped
                }
                if v1_screen.x < -guardband || v1_screen.x > guardband || v1_screen.y < -guardband || v1_screen.y > guardband {
                    self.stats.push_stat("clipped by guardband", "v1");
                    continue; // clipped
                }
                if v2_screen.x < -guardband || v2_screen.x > guardband || v2_screen.y < -guardband || v2_screen.y > guardband {
                    self.stats.push_stat("clipped by guardband", "v2");
                    continue; // clipped
                }

                let v0_rasterizer = VertexOutput {
                    position: v0.position,
                    normal: normal,
                    color: v0.color,
                    texcoord: v0.texcoord,
                    uv_lightmap: v0.uv_lightmap,
                    screen_position: v0_screen,
                    world_position: v0.world_position,
                };

                let v1_rasterizer = VertexOutput {
                    position: v1.position,
                    normal: normal,
                    color: v1.color,
                    texcoord: v1.texcoord,
                    uv_lightmap: v1.uv_lightmap,
                    screen_position: v1_screen,
                    world_position: v1.world_position,
                };

                let v2_rasterizer = VertexOutput {
                    position: v2.position,
                    normal: normal,
                    color: v2.color,
                    texcoord: v2.texcoord,
                    uv_lightmap: v2.uv_lightmap,
                    screen_position: v2_screen,
                    world_position: v2.world_position,
                };

                // Send to the rasterizer
                if self.settings.naive_rasterization {
                    self.rasterize_triangle_naive(&v2_rasterizer, &v1_rasterizer, &v0_rasterizer, texture, &lightmap, &lightmap_size);
                }
                else {
                    unreachable!();
                    // self.rasterize_triangle(&v2_rasterizer, &v1_rasterizer, &v0_rasterizer);
                }
            }
        }
    }

    fn texture_sample(texture: &Texture, uv: Vec2) -> Color {
        let mip_level = 0;

        let mip_width: u32 = texture.width >> mip_level;
        let mip_height: u32 = texture.height >> mip_level;

        let wrap_uv = uv - uv.floor();

        let mip_level_data: &Vec<u8> = &texture.data[mip_level];
        let coords = ivec2(
            (wrap_uv.x * (mip_width - 1) as f32) as i32,
            (wrap_uv.y * (mip_height - 1) as f32) as i32,
        );

        let indexed_color: u8 = mip_level_data[(coords.y as u32 * mip_width + coords.x as u32) as usize];
        Color::from_u32(texture.palette[indexed_color as usize])
    }

    fn sample_lightmap_point(data: &[u8], size: &IVec2, uv: &Vec2) -> u8 {
        let coords = ivec2(
            (uv.x * (size.x - 1) as f32) as i32,
            (uv.y * (size.y - 1) as f32) as i32,
        );

        data[(coords.y * size.x + coords.x) as usize]
    }

    fn sample_lightmap_bilinear(data: &[u8], size: &IVec2, uv: &Vec2) -> u8 {
        // let clamp_uv = uv - uv.floor();
        let clamp_uv = uv.clamp(Vec2::splat(0.001), Vec2::splat(0.999));
        // let clamp_uv = uv;

        // Convert UV to lightmap space (texel indices)
        let x = clamp_uv.x * (size.x - 1) as f32;
        let y = clamp_uv.y * (size.y - 1) as f32;

        // Compute integer texel coordinates (clamping to avoid out-of-bounds)
        let x0 = x.floor() as i32;
        let y0 = y.floor() as i32;
        let x1 = (x0 + 1).clamp(0, size.x - 1);
        let y1 = (y0 + 1).clamp(0, size.y - 1);

        // Get fractional parts for interpolation
        let wu = x - x0 as f32;
        let wv = y - y0 as f32;

        // Fetch the four nearest texels (handling edge cases by clamping)
        let i00 = (y0 * size.x + x0) as usize;
        let i10 = (y0 * size.x + x1) as usize;
        let i01 = (y1 * size.x + x0) as usize;
        let i11 = (y1 * size.x + x1) as usize;

        let l00 = data[i00] as f32 / 255.0;
        let l10 = data[i10] as f32 / 255.0;
        let l01 = data[i01] as f32 / 255.0;
        let l11 = data[i11] as f32 / 255.0;

        // Bilinear interpolation
        let lerp_x0 = l00 * (1.0 - wu) + l10 * wu;
        let lerp_x1 = l01 * (1.0 - wu) + l11 * wu;
        let final_light = lerp_x0 * (1.0 - wv) + lerp_x1 * wv;

        (final_light * 255.0) as u8
    }

    fn pixel_shader(v: &VertexOutput, texture: &Texture, lightmap: &[u8], lightmap_size: &IVec2) -> Vec4 {
        let tex = Self::texture_sample(texture, v.texcoord).get_vec();

        let mut light: Vec3 = Vec3::ZERO;

        if lightmap.len() > 0 {
            let brightness = Self::sample_lightmap_bilinear(&lightmap, &lightmap_size, &v.uv_lightmap);
            light = vec3(brightness as f32 / 255.0, brightness as f32 / 255.0, brightness as f32 / 255.0); // - v.color.xyz();
        }
        else {
            light = Vec3::ONE - v.color.xyz();
        }

        (tex.xyz() * light).extend(1.0)
        // tex.xyz().extend(1.0)
    }

    fn edge_function(v0: IVec2, v1: IVec2, p: IVec2) -> i32 {
        (v1.x - v0.x) * (p.y - v0.y) - (v1.y - v0.y) * (p.x - v0.x)
    }

    /*fn rasterize_triangle(&mut self, v0: &VertexOutput, v1: &VertexOutput, v2: &VertexOutput) {
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
                                tex_coord: interpolated_tex_coord,
                                normal: interpolated_normal,
                                world_position: Vec3::ZERO,
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
                                    world_position: Vec3::ZERO,
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
    }*/

    fn rasterize_triangle_naive(
        &mut self,
        v0: &VertexOutput,
        v1: &VertexOutput,
        v2: &VertexOutput,
        texture: &Texture,
        lightmap: &[u8],
        lightmap_size: &IVec2,
    ) {
        self.stats.rasterizer_input += 1;

        let iv0 = IVec2::new(v0.screen_position.x.ceil() as i32, v0.screen_position.y.ceil() as i32);
        let iv1 = IVec2::new(v1.screen_position.x.ceil() as i32, v1.screen_position.y.ceil() as i32);
        let iv2 = IVec2::new(v2.screen_position.x.ceil() as i32, v2.screen_position.y.ceil() as i32);

        // compute bounding box
        let mut min_x: i32 = iv0.x.min(iv1.x).min(iv2.x);
        let mut min_y: i32 = iv0.y.min(iv1.y).min(iv2.y);
        let mut max_x: i32 = iv0.x.max(iv1.x).max(iv2.x);
        let mut max_y: i32 = iv0.y.max(iv1.y).max(iv2.y);

        // clip it against screen
        min_x = min_x.max(0);
        min_y = min_y.max(0);
        max_x = max_x.min(self.screen_width as i32 - 1);
        max_y = max_y.min(self.screen_height as i32 - 1);
        
        // calculate the area of the triangle (2x)
        let area: i32 = Self::edge_function(iv0, iv1, iv2);
        if area <= 0 { // negative area means it's a back-facing triangle
            return;
        }

        // pre-calculate the area for the first pixel in the grid
        // let mut w0_min = Self::edge_function(iv1, iv2, bb_min);
        // let mut w1_min = Self::edge_function(iv2, iv0, bb_min);
        // let mut w2_min = Self::edge_function(iv0, iv1, bb_min);

        // and calculate steps needed to move the barycentric coordinates
        // let w0_stepx = iv1.y - iv2.y;
        // let w0_stepy = iv2.x - iv1.x;
        // let w1_stepx = iv2.y - iv0.y;
        // let w1_stepy = iv0.x - iv2.x;
        // let w2_stepx = iv0.y - iv1.y;
        // let w2_stepy = iv1.x - iv0.x;

        // Compute perspective-correct inverse depth
        let inv_z0 = 1.0 / v0.screen_position.z;
        let inv_z1 = 1.0 / v1.screen_position.z;
        let inv_z2 = 1.0 / v2.screen_position.z;

        // Pre-divide texcoords by depth
        let texcoord0 = v0.texcoord / v0.screen_position.w;
        let texcoord1 = v1.texcoord / v1.screen_position.w;
        let texcoord2 = v2.texcoord / v2.screen_position.w;

        let uv_lightmap0 = v0.uv_lightmap / v0.screen_position.w;
        let uv_lightmap1 = v1.uv_lightmap / v1.screen_position.w;
        let uv_lightmap2 = v2.uv_lightmap / v2.screen_position.w;

        for y in min_y..=max_y {
            // initialize barycentric coords
            // let mut w0 = w0_min;
            // let mut w1 = w1_min;
            // let mut w2 = w2_min;

            for x in min_x..=max_x {
                // let p = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, 0.0);
                let ip = IVec2::new(x, y);

                let w0 = Self::edge_function(iv1, iv2, ip);
                let w1 = Self::edge_function(iv2, iv0, ip);
                let w2 = Self::edge_function(iv0, iv1, ip);

                // skip if outside the triangle
                if (w0 | w1 | w2) >= 0 {
                    let pixel_index: usize = y as usize * self.screen_width + x as usize;

                    // normalize our lambdas and move to floating point math
                    let l0 = w0 as f32 / area as f32;
                    let l1 = w1 as f32 / area as f32;
                    let l2 = w2 as f32 / area as f32;

                    // calculate perspective-correct depth
                    let pixel_z = 1.0 / (inv_z0 * l0 + inv_z1 * l1 + inv_z2 * l2);

                    // test against our z-buffer
                    let zbuffer_value = self.depth_buffer[pixel_index];
                    if pixel_z > zbuffer_value {
                        continue;
                    }

                    let inv_w = (l0 / v0.position.w + l1 / v1.position.w + l2 / v2.position.w);
                    let pixel_w = 1.0 / inv_w;

                    // now we need to calculate the interpolated vertex attributes... fun!
                    let interpolated_screen: Vec4 = Vec4::new(x as f32 + 0.5, y as f32 + 0.5, pixel_z, 0.0);
                    let interpolated_position = l0 * v0.position + l1 * v1.position + l2 * v2.position;
                    let interpolated_normal = l0 * v0.normal + l1 * v1.normal + l2 * v2.normal;
                    let interpolated_texcoord = (l0 * texcoord0 + l1 * texcoord1 + l2 * texcoord2) * pixel_w;
                    let interpolated_uv_lightmap = (l0 * uv_lightmap0 + l1 * uv_lightmap1 + l2 * uv_lightmap2) * pixel_w;
                    let interpolated_color = l0 * v0.color + l1 * v1.color + l2 * v2.color;

                    let v = VertexOutput {
                        screen_position: interpolated_screen,
                        position: interpolated_position,
                        color: interpolated_color,
                        texcoord: interpolated_texcoord,
                        uv_lightmap: interpolated_uv_lightmap,
                        normal: interpolated_normal,
                        world_position: Vec3::ZERO,
                    };

                    // inc stats
                    self.stats.pixels_drawn += 1;

                    // run the "pixel shader" for this pixel
                    let c = Self::pixel_shader(&v, texture, lightmap, lightmap_size);

                    self.back_buffer[pixel_index] = Color::from_vec4(c).to_u32();
                    self.depth_buffer[pixel_index] = pixel_z;
                }

                // take a step to the right
                // w0 += w0_stepx;
                // w1 += w1_stepx;
                // w2 += w2_stepx;
            }

            // and a step down
            // w0_min += w0_stepy;
            // w1_min += w1_stepy;
            // w2_min += w2_stepy;
        }
    }
}
