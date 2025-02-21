use crate::renderer::Renderer;
use glam::Mat4;
use std::fs::File;
use std::io;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug)]
pub struct Model {
    pub vertices: Vec<glam::Vec3>,
    pub indices: Vec<u32>,
}

impl Model {
    pub(crate) fn load<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for line in reader.lines() {
            let line = line?;
            let tokens: Vec<&str> = line.split_whitespace().collect();

            if tokens.is_empty() {
                continue;
            }

            match tokens[0] {
                "v" => {
                    if tokens.len() < 4 {
                        continue;
                    }

                    let v = glam::Vec3::new(
                        tokens[1].parse::<f32>().unwrap_or(0.0),
                        tokens[2].parse::<f32>().unwrap_or(0.0),
                        tokens[3].parse::<f32>().unwrap_or(0.0),
                    );

                    vertices.push(v);
                }
                "f" => {
                    if tokens.len() < 4 {
                        continue;
                    }

                    for i in 1..tokens.len() {
                        let index = tokens[i].split('/')
                            .next()
                            .unwrap_or("0")
                            .parse::<u32>()
                            .unwrap_or(1) - 1;
                        indices.push(index);
                    }
                }
                _ => {
                    // ignore unknown lines
                }
            }
        }

        Ok(Self { vertices, indices })
    }

    fn draw(model: &Model, w: &Mat4, wvp: &Mat4, renderer: &mut Renderer) {
        // for tri in model.indices.chunks_exact(3) {
        //     let mut vertices: [Vertex; 3] = [Vertex::default(); 3];
        // 
        //     // temp: calculate normal for triangle
        //     let p0: Vec3 = model.vertices[tri[0] as usize];
        //     let p1: Vec3 = model.vertices[tri[1] as usize];
        //     let p2: Vec3 = model.vertices[tri[2] as usize];
        //     let normal = ((p1 - p0).cross(p2 - p0)).normalize();
        // 
        //     let mut visible = false;
        // 
        //     for i in 0..3 {
        //         let obj_position: Vec3 = model.vertices[tri[i] as usize];
        //         let obj_normal: Vec3 = normal;
        // 
        //         let s_position = wvp.project_point3(obj_position);
        //         if !visible && s_position.x > -1.0 && s_position.x < 1.0 && s_position.y > -1.0 && s_position.y < 1.0 {
        //             visible = true;
        //         }
        // 
        //         let w_position = w.transform_point3(obj_position);
        //         let w_normal = w.transform_vector3(obj_normal); // leaving this here since in theory this should be different per-vertex and come from the 3d mesh
        // 
        //         let p_screen = IVec2::new(
        //             ((s_position.x + 1.0) * 0.5 * SCREEN_WIDTH as f32) as i32,
        //             ((1.0 - s_position.y) * 0.5 * SCREEN_HEIGHT as f32) as i32,
        //         );
        // 
        //         let vertex_input = Vertex {
        //             screen_position: p_screen,
        //             position: w_position,
        //             normal: w_normal,
        //         };
        // 
        //         vertices[i] = vertex_input;
        //     }
        // 
        //     if !visible { // all 3 vertices are off-screen
        //         continue;
        //     }
        // 
        //     draw_triangle(vertices[0], vertices[1], vertices[2], buffer);
        // }
    }
}