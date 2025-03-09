use crate::backbuffer::BackBuffer;
use crate::bsp::{BSPClipNode, BSPEdge, BSPFace, BSPHeader, BSPLeaf, BSPLumps, BSPModel, BSPNode, BSPPlane, BSPSurface};
use crate::engine::{DebugStats, Engine};
use crate::entity::{Entity, EntityData, InfoEntity, MoveType, SolidType};
use crate::game::doors::FuncDoor;
use crate::game::triggers::{TriggerMultiple, TriggerOnce};
use crate::math::{BBoxShort, BoundBox, Plane};
use crate::renderer::{Texture, Vertex};
use crate::{bsp, entity};
use glam::{ivec2, vec2, vec3, vec4, Mat4, Vec2, Vec3, Vec4};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::rc::Rc;
use std::{io, ptr};

enum ModelType {
    Brush,
    Sprite,
    Alias
}

enum ContentsType {
    Empty = -1,
    Solid = -2,
    Water = -3,
    Slime = -4,
    Lava = -5,
    Sky = -6,
    Origin = -7,
    Clip = -8,
}

#[derive(Debug, Default)]
pub struct HitResult {
    pub all_solid: bool,
    pub start_solid: bool,
    pub plane: Plane, // TODO: replace with normal ?
    pub fraction: f32,
    pub end_position: Vec3,
}

const LIGHT_ANIMATIONS: [&'static [u8]; 11] = [
    b"m", // normal
    b"mmnmmommommnonmmonqnmmo", // Flicker 1
    b"abcdefghijklmnopqrstuvwxyzyxwvutsrqponmlkjihgfedcba", //Slow strong pulse
    b"mmmmmaaaaammmmmaaaaaabcdefgabcdefg", //Candle 1
    b"mamamamamama", //Fast Strobe
    b"jklmnopqrstuvwxyzyxwvutsrqponmlkj", //Gentle Pulse
    b"nmonqnmomnmomomno", //Flicker 2
    b"mmmaaaabcdefgmmmmaaaammmaamm", //Candle 2
    b"mmmaaammmaaammmabcdefaaaammmmabcdefmmmaaaa", //Candle 3
    b"aaaaaaaazzzzzzzz", //Slow strobe 4
    b"mmamammmmammamamaaamammma", //Fluorescent Flicker
];

pub struct Model {
    pub bound: BoundBox,
    pub origin: Vec3,
    pub numleafs: i32,
    pub face_id: i32,
    pub face_num: i32,
    pub hulls: Vec<Hull>,
}

#[derive(Clone)]
pub struct Hull {
    pub clip_nodes: Rc<Box<[BSPClipNode]>>,
    pub planes: Rc<Box<[BSPPlane]>>,
    pub first_clip_node: i32,
    pub last_clip_node: i32,
    pub clip_mins: Vec3,
    pub clip_maxs: Vec3,
}

pub struct BSPData {
    planes: Rc<Box<[BSPPlane]>>,
    vertices: Box<[Vec3]>,
    visibility: Vec<u8>,
    nodes: Vec<BSPNode>,
    surfaces: Vec<BSPSurface>,
    faces: Vec<BSPFace>,
    lightmaps: Vec<u8>,
    clip_nodes: Rc<Box<[BSPClipNode]>>,
    leafs: Vec<BSPLeaf>,
    list_faces: Vec<u16>,
    edges: Vec<BSPEdge>,
    list_edges: Vec<i32>,
    submodels: Vec<BSPModel>,
    textures: Vec<Texture>,
    entities: Vec<HashMap<String, String>>,
}

pub struct Level {
    data: BSPData,

    current_leaf_index: u16,
    visible_leafs: Vec<u16>,

    textures_map: HashMap<String, u16>,

    models: Vec<Model>,
    entities: Vec<(EntityData, Box<dyn Entity>)>,

    creators: HashMap<String, Box<dyn Fn() -> Box<dyn Entity>>>,

    hull0_clip_nodes: Rc<Box<[BSPClipNode]>>,
}

impl Level {
    pub fn load_bsp(file: &mut File) -> io::Result<BSPData> {
        // read header
        let mut buffer = vec![0u8; size_of::<BSPHeader>()];
        file.read_exact(&mut buffer)?;
        let bsp_header = unsafe { ptr::read(buffer.as_ptr() as *const BSPHeader) };

        if bsp_header.version != 29 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "unsupported bsp version"));
        }

        // read bsp nodes
        let entities_desc = bsp::read_entities(file, &bsp_header, BSPLumps::Entities).expect("Failed to read entities");
        let planes: Vec<BSPPlane> = bsp::read_lump(file, &bsp_header, BSPLumps::Planes).expect("Failed to read BSP planes");
        let textures: Vec<Texture> = bsp::read_textures(file, &bsp_header, BSPLumps::Textures).expect("Failed to read textures");
        let vertices: Vec<Vec3> = bsp::read_lump(file, &bsp_header, BSPLumps::Vertices).expect("Failed to read vertices");
        let visibility: Vec<u8> = bsp::read_lump(file, &bsp_header, BSPLumps::Visibility).expect("Failed to read vislist");
        let nodes: Vec<BSPNode> = bsp::read_lump(file, &bsp_header, BSPLumps::Nodes).expect("Failed to read BSP nodes");
        let texture_infos: Vec<BSPSurface> = bsp::read_lump(file, &bsp_header, BSPLumps::TexInfo).expect("Failed to read texture infos");
        let faces: Vec<BSPFace> = bsp::read_lump(file, &bsp_header, BSPLumps::Faces).expect("Failed to read BSP faces");
        let lightmaps: Vec<u8> = bsp::read_lump(file, &bsp_header, BSPLumps::Lighting).expect("Failed to read lightmaps");
        let clip_nodes: Vec<BSPClipNode> = bsp::read_lump(file, &bsp_header, BSPLumps::ClipNodes).expect("Failed to read clip nodes");
        let leafs: Vec<BSPLeaf> = bsp::read_lump(file, &bsp_header, BSPLumps::Leafs).expect("Failed to read BSP leafs");
        let list_faces: Vec<u16> = bsp::read_lump(file, &bsp_header, BSPLumps::MarkSurfaces).expect("Failed to read BSP list of faces");
        let edges: Vec<BSPEdge> = bsp::read_lump(file, &bsp_header, BSPLumps::Edges).expect("Failed to read BSP edges");
        let list_edges: Vec<i32> = bsp::read_lump(file, &bsp_header, BSPLumps::SurfEdges).expect("Failed to read BSP list of edges");
        let submodels: Vec<BSPModel> = bsp::read_lump(file, &bsp_header, BSPLumps::Models).expect("bsp model load failed");

        Ok(BSPData {
            entities: entities_desc,
            planes: Rc::new(planes.into_boxed_slice()),
            vertices: vertices.into_boxed_slice(),
            visibility,
            nodes,
            surfaces: texture_infos,
            faces,
            lightmaps,
            clip_nodes: Rc::new(clip_nodes.into_boxed_slice()),
            leafs,
            list_faces,
            edges,
            list_edges,
            submodels,

            textures, // remove
        })
    }

    pub fn load<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        // the original quake code basically copies the model_t struct for each submodel, copies the
        // same pointers over and just updates the values that change (like face start + faces count, etc).
        // Not only I dont like that code much (sorry John) in rust we also want to be more careful with data ownership

        // I'm pretty sure this was done so all models (BSP, Alias, or Sprites) had the same representation
        // and the engine simply treats them all the same, and can find them by name (in case of
        // bps models I know that they start with an *)... so we'll see how we handle that once we
        // implement alias models.

        let mut file: File = File::open(path)?;
        let data = Self::load_bsp(&mut file)?;

        let mut textures_map: HashMap<String, u16> = HashMap::new();
        for i in 0..data.textures.len() {
            let texture = &data.textures[i];
            textures_map.insert(texture.name.clone(), i as u16);
        }

        // build hull 0 from the rendering bsp nodes
        let hull0_clip_nodes: Vec<BSPClipNode> = data.nodes.iter().map(|node| {
            let extract_leaf_type = |index: u16| {
                if index & 0x8000 != 0 {
                    data.leafs[(!index) as usize].leaf_type as i16
                } else {
                    index as i16
                }
            };

            BSPClipNode {
                plane_id: node.plane_id as u32,
                front: extract_leaf_type(node.front),
                back: extract_leaf_type(node.back),
            }
        }).collect();

        let mut level = Self {
            data,
            entities: Vec::new(),
            current_leaf_index: 0,
            visible_leafs: vec![],
            textures_map,
            creators: HashMap::new(),
            models: vec![],
            hull0_clip_nodes: Rc::new(hull0_clip_nodes.into_boxed_slice()),
        };

        // Define all models
        let models: Vec<Model> = level.data.submodels.iter().map(|submodel| {
            let hulls: Vec<_> = (0..3).map(|hull_index| {
                let (clip_mins, clip_maxs) = match hull_index {
                    0 => (vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0)),
                    1 => (vec3(-16.0, -16.0, -24.0), vec3(16.0, 16.0, 32.0)),
                    2 => (vec3(-32.0, -32.0, -24.0), vec3(32.0, 32.0, 64.0)),
                    _ => unreachable!(),
                };

                let (first_clip_node, last_clip_node, clip_nodes) = if hull_index == 0 {
                    (
                        submodel.head_node[0],
                        level.hull0_clip_nodes.len() as i32 - 1,
                        level.hull0_clip_nodes.clone(),
                    )
                } else {
                    (
                        submodel.head_node[hull_index as usize],
                        level.data.clip_nodes.len() as i32 - 1,
                        level.data.clip_nodes.clone(),
                    )
                };

                Hull {
                    clip_nodes,
                    planes: level.data.planes.clone(),
                    first_clip_node,
                    last_clip_node,
                    clip_mins,
                    clip_maxs,
                }
            }).collect();

            Model {
                bound: submodel.bound,
                origin: submodel.origin,
                numleafs: submodel.numleafs,
                face_id: submodel.face_id,
                face_num: submodel.face_num,
                hulls,
            }
        }).collect();

        // Register our entity types
        // TODO: This belongs in game... and probably need to be moved outside level
        level.register::<FuncDoor>("func_door");
        level.register::<TriggerOnce>("trigger_once");
        level.register::<TriggerMultiple>("trigger_multiple");

        let entities_desc = level.data.entities.clone();
        for entity_desc in entities_desc {
            level.create_entity(&entity_desc);
        }

        level.models = models;

        Ok(level)
    }

    pub fn get_entity_of_class_name(&self, name: &str) -> Option<&EntityData> {
        for (entity, _behaviour) in &self.entities {
            if entity.class_name == name {
                return Some(entity);
            }
        }

        None
    }

    pub fn register<T: Entity + Default + 'static>(&mut self, classname: &str) {
        self.creators.insert(
            classname.to_string(),
            Box::new(|| Box::new(T::default())),
        );
    }

    pub fn create_entity(&mut self, properties: &HashMap<String, String>) {
        let class_name = &properties["classname"];

        if class_name == "light" { // ignore lights for now
            return;
        }

        let mut entity = EntityData {
            class_name: class_name.clone(),
            origin: entity::parse_vec3(properties, "origin").unwrap_or(Vec3::ZERO),
            solid: SolidType::Not,
            angle: entity::parse_f32(properties, "angle").unwrap_or(0.0),
            model_index: entity::parse_model(properties, "model").unwrap_or(-1),
            move_type: MoveType::None,
        };
        let behaviour = self.creators
            .get(class_name.as_str())
            .map(|creator| creator());

        println!("Created entity: {:?}", class_name);

        if let Some(mut behaviour) = behaviour {
            behaviour.construct(&mut entity);
            self.entities.push((entity, behaviour));
        }
        else {
            self.entities.push((entity, Box::new(InfoEntity::default())));
        }
    }

    pub fn update_visibility(&mut self, position: Vec3) {
        let current_leaf_index = self.find_leaf(0, position);
        if current_leaf_index == self.current_leaf_index || current_leaf_index == 0 {
            return;
        }

        let start_leaf = &self.data.leafs[current_leaf_index as usize];

        self.visible_leafs = vec![];
        let num_leaves = self.data.leafs.len() as u16;

        // leaf.vislist marks the offset where the visibility list for this leaf starts
        let mut v = start_leaf.vislist as usize;

        let mut l: u16 = 1;
        while l < num_leaves
        {
            if self.data.visibility[v] == 0
            {
                // if we read a 0, the next byte tells us how many bytes to skip (RLE)
                // each bit represents a leaf, so we skip that amount of leaf indices (L)
                l += 8 * self.data.visibility[v + 1] as u16;
                v += 1;
            }
            else
            {
                // tag the 8 leafs in this byte
                for bit in 0..=7 {
                    if self.data.visibility[v] & (1u8 << bit) != 0 {
                        if l < num_leaves {
                            self.visible_leafs.push(l);
                        }
                    }
                    l += 1;
                }
            }

            v += 1;
        }
    }

    pub fn draw(&self, engine: &Engine, stats: &mut DebugStats, back_buffer: &mut BackBuffer, position: Vec3, w: &Mat4, v: &Mat4, p: &Mat4, wvp: &Mat4) {
        let mut drawn_faces: Vec<u16> = vec![];

        let frustum_planes = Plane::extract_frustum_planes_world_space(&p, &v);

        // Collect visible leafs
        let node = &self.data.nodes[0];
        let mut visible_leafs: Vec<u16> = vec![];
        self.traverse_bsp_tree(engine, stats, back_buffer, node, position, &frustum_planes, w, wvp, &mut drawn_faces, &mut visible_leafs);

        stats.leafs_visible += visible_leafs.len() as u32;

        // Now collect faces, faces can be repeated in different leafs (AFAIK)
        let mut visible_faces: Vec<u16> = vec![];
        for leaf_index in visible_leafs {
            let leaf: &BSPLeaf = &self.data.leafs[leaf_index as usize];

            let face_list_offset = leaf.lface_id;
            let face_list_num = face_list_offset + leaf.lface_num;

            for face_list_index in face_list_offset..face_list_num {
                let face_index: u16 = self.data.list_faces[face_list_index as usize];
                if !visible_faces.contains(&face_index) {
                    visible_faces.push(face_index);
                }
            }
        };

        stats.faces_rendered += visible_faces.len() as u32;

        // Draw faces
        self.draw_faces(engine, stats, back_buffer, &visible_faces, w, wvp);

        // DEBUG: Draw all leafs
        // self.bsp_leafs.iter().for_each(|l| {
             // self.draw_leaf(engine, stats, back_buffer, &l, &frustum_planes, &mut drawn_faces, w, wvp);
        // });

        // Draw entities
        for (entity, _behaviour) in &self.entities
        {
            if entity.solid != SolidType::Trigger {
                let model_index = entity.model_index;
                if model_index >= 1 {
                    // let model: &BSPModel = &self.models[model_index as usize];
                    let model: &Model = &self.models[model_index as usize];
                    self.draw_model(model, engine, back_buffer, &frustum_planes, w, wvp, stats);
                }
            }
        }
    }

    fn draw_model(&self, model: &Model, engine: &Engine, back_buffer: &mut BackBuffer, frustum_planes: &[Plane;6], w: &Mat4, wvp: &Mat4, stats: &mut DebugStats) {
        // Draw all submodels
        let mut model_faces: Vec<u16> = vec![];

        if Self::is_aabb_outside_frustum(&frustum_planes, &model.bound) {
            return;
        }

        stats.models_rendered += 1;

        let range_start: u16 = model.face_id as u16;
        let range_end: u16 = (model.face_id + model.face_num) as u16;
        for face_index in range_start..range_end {
            if !model_faces.contains(&face_index) {
                model_faces.push(face_index);
            }
        }

        self.draw_faces(engine, stats, back_buffer, &model_faces, w, wvp);
    }

    fn traverse_bsp_tree(
        &self,
        engine: &Engine,
        stats: &mut DebugStats,
        back_buffer: &mut BackBuffer,
        node: &BSPNode,
        position: Vec3,
        frustum_planes: &[Plane; 6],
        w: &Mat4,
        wvp: &Mat4,
        drawn_faces: &mut Vec<u16>,
        visited_leafs: &mut Vec<u16>,
    )
    {
        stats.bsp_nodes_traversed += 1;

        // Stop processing node branch if outside our frustum
        if Self::is_aabb_outside_frustum_short(frustum_planes, &node.bound) {
            return;
        }

        let plane = &self.data.planes[node.plane_id as usize];
        let distance: f32 = plane.normal.dot(position) - plane.distance;

        let children = if distance >= 0.0 {
            [node.front, node.back]
        }
        else {
            [node.back, node.front]
        };

        for child in children {
            if child & 0x8000 == 0 {
                let first_node = &self.data.nodes[child as usize];
                self.traverse_bsp_tree(engine, stats, back_buffer, first_node, position, frustum_planes, w, wvp, drawn_faces, visited_leafs);
            } else {
                let leaf_index = !child;

                if self.visible_leafs.contains(&leaf_index) {
                    let leaf = &self.data.leafs[leaf_index as usize];

                    if Self::is_aabb_outside_frustum_short(frustum_planes, &leaf.bound) {
                        continue;
                    }

                    visited_leafs.push(leaf_index);
                }
            }
        }
    }

    fn get_texture(&self, name: &String) -> Option<&Texture> {
        self.textures_map.get(name).and_then(|&index| self.data.textures.get(index as usize))
    }


    fn find_leaf(&self, node_index: u16, position: Vec3) -> u16 {
        let mut n = node_index;
        while n & 0x8000 == 0 {
            let node = &self.data.nodes[n as usize];
            let plane = &self.data.planes[node.plane_id as usize];

            let distance: f32 = plane.normal.dot(position) - plane.distance;

            if distance >= 0.0 {
                n = node.front
            }
            else {
                n = node.back
            }
        }

        !n
    }

    fn find_face_dimensions(&self, face: &BSPFace) -> (f32, f32, f32, f32) {
        let mut min_u: f32 = f32::MAX;
        let mut min_v: f32 = f32::MAX;
        let mut max_u: f32 = f32::MIN;
        let mut max_v: f32 = f32::MIN;

        let tex_info: &BSPSurface = &self.data.surfaces[face.texinfo_id as usize];

        for edge_list_index in face.ledge_id..(face.ledge_id + (face.ledge_num as i32)) {
            let edge_index = self.data.list_edges[edge_list_index as usize];

            let vertex: Vec3;
            if edge_index >= 0 {
                let edge = &self.data.edges[edge_index as usize];
                vertex = self.data.vertices[edge.vertex0 as usize];
            } else {
                let edge = &self.data.edges[-edge_index as usize];
                vertex = self.data.vertices[edge.vertex1 as usize];
            }

            let u = vertex.dot(tex_info.u_axis) + tex_info.u_offset;
            let v = vertex.dot(tex_info.v_axis) + tex_info.v_offset;

            let floor_u = u.floor();
            let floor_v = v.floor();

            min_u = min_u.min(floor_u);
            min_v = min_v.min(floor_v);
            max_u = max_u.max(floor_u);
            max_v = max_v.max(floor_v);
        }

        (min_u, min_v, max_u, max_v)
    }

    fn animate_texture_coordinates(vertex: &Vec3, uv_texture: &Vec2, time: f32) -> Vec2 {
        let frequency = 0.5;
        let amplitude = 0.05;
        let speed = 1.0;

        let wave_x = (vertex.x * frequency + time * speed).sin();
        let wave_y = (vertex.y * frequency + time * speed).sin();

        vec2(
            uv_texture.x + wave_x * amplitude,
            uv_texture.y + wave_y * amplitude
        )
    }

    fn draw_faces(
        &self,
        engine: &Engine,
        stats: &mut DebugStats,
        back_buffer: &mut BackBuffer,
        faces: &Vec<u16>,
        w: &Mat4,
        wvp: &Mat4
    ) {
        let time = engine.time().time_since_start();

        for face_index in faces {
            let face: &BSPFace = &self.data.faces[*face_index as usize];
            let tex_info: &BSPSurface = &self.data.surfaces[face.texinfo_id as usize];
            let mut texture: &Texture = &self.data.textures[tex_info.texture_id as usize];

            if texture.frames > 1 {
                let frame = ((time / 0.2) as i32) % texture.frames as i32;

                let actual_name = &texture.name[2..]; // "texture"
                let frame_name = format!("+{}{}", frame, actual_name);

                texture = self.get_texture(&frame_name).expect(format!("Failed to get texture frame {:?} {:?}", frame_name, texture.frames).as_str());
            }

            // Calculate min u/v and max u/v
            let (min_u, min_v, max_u, max_v) = self.find_face_dimensions(face);

            // Calculate lightmap size
            let lightmap_size = ivec2(
                ((max_u / 16.0).ceil() - (min_u / 16.0).floor() + 1.0) as i32,
                ((max_v / 16.0).ceil() - (min_v / 16.0).floor() + 1.0) as i32
            );

            let mut face_vertices: Vec<Vertex> = vec![];
            let mut face_indices : Vec<u32> = Vec::new();

            for edge_list_index in face.ledge_id..(face.ledge_id + (face.ledge_num as i32)) {
                let edge_index = self.data.list_edges[edge_list_index as usize];

                let vertex: Vec3;
                if edge_index >= 0 {
                    let edge = &self.data.edges[edge_index as usize];
                    vertex = self.data.vertices[edge.vertex0 as usize];
                }
                else {
                    let edge = &self.data.edges[-edge_index as usize];
                    vertex = self.data.vertices[edge.vertex1 as usize];
                }

                let u = vertex.dot(tex_info.u_axis) + tex_info.u_offset;
                let v = vertex.dot(tex_info.v_axis) + tex_info.v_offset;

                let is_water = texture.name.starts_with('*');

                // Pick uv coordinates or warp them if this is a liquid surface
                let uv_texture: Vec2 = if is_water {
                    // let uv = vec2(u / texture.width as f32, v / texture.height as f32);
                    // let uv = vec2(u, v);
                    // let uv2 = Self::animate_texture_coordinates(&vertex, &uv, time);
                    // vec2(uv2.x / texture.width as f32, uv2.y / texture.height as f32)

                    let uv = vec2(u / texture.width as f32, v / texture.height as f32);
                    Self::animate_texture_coordinates(&vertex, &uv, time)
                }
                else {
                    vec2(u / texture.width as f32, v / texture.height as f32)
                };

                let base_light = if is_water {
                    127
                }
                else {
                    face.baselight
                };

                // pick color (light color) from the light type
                let color: Vec4 = match face.typelight {
                    0..=10 => {
                        let c = 1.0 - self.get_light_intensity(engine.time().time_since_start(), face.typelight);
                        vec4(c, c, c, 1.0)
                    }
                    255 => {
                        Vec3::splat(1.0 - (base_light as f32) / 255.0).extend(1.0)
                    }
                    32..62 => {
                        // programmable lights
                        vec4(0.0, 0.0, 0.0, 1.0)
                    }
                    _ => {
                        // println!("unhandled {:?}", face.typelight);

                        vec4(0.0, 0.0, 0.0, 1.0)
                    }
                };

                let uv_lightmap = vec2(
                    (u - min_u) / (max_u - min_u),
                    (v - min_v) / (max_v - min_v),
                );

                face_vertices.push(Vertex {
                    position: vertex,
                    normal: Vec3::ZERO,
                    color,
                    tex_coord: uv_texture,
                    uv_lightmap,
                });
            }

            for i in 2..face_vertices.len() {
                face_indices.push(0);

                face_indices.push(i as u32);
                face_indices.push((i - 1) as u32);
            }

            if face.lightmap != -1 {
                let lightmap_data: &[u8] = &self.data.lightmaps[face.lightmap as usize..(face.lightmap + lightmap_size.x * lightmap_size.y) as usize];
                engine.renderer().draw(stats, back_buffer, &face_vertices, &face_indices, w, wvp, texture, lightmap_data, &lightmap_size);
            }
            else {
                let lightmap_data: &[u8] = &[]; // Declare slice reference
                let lightmap_size = ivec2(0,0);

                engine.renderer().draw(stats, back_buffer, &face_vertices, &face_indices, w, wvp, texture, lightmap_data, &lightmap_size);
            }
        }
    }

    fn get_light_intensity(&self, elapsed_time: f32, light_type: u8) -> f32 {
        let frames = LIGHT_ANIMATIONS[light_type as usize];

        let index = (elapsed_time / 0.1) as usize % frames.len();
        let c = frames[index];

        let level = c.saturating_sub(b'a') as f32;  // 0 to 25
        1.0 - level / 25.0
    }

    fn is_aabb_outside_frustum_short(frustum_planes: &[Plane; 6], bounds: &BBoxShort) -> bool {
        for plane in frustum_planes {
            if Plane::is_bboxshort_outside_plane(&plane, &bounds) {
                return true;
            }
        }

        false
    }

    fn is_aabb_outside_frustum(frustum_planes: &[Plane; 6], bounds: &BoundBox) -> bool {
        for plane in frustum_planes {
            if Plane::is_bbox_outside_plane(&plane, bounds) {
                return true;
            }
        }

        false
    }

    fn plane_distance(plane: &BSPPlane, p: &Vec3) -> f32 {
        if plane.plane_type < 3 {
            p[plane.plane_type as usize] - plane.distance
        }
        else {
            plane.normal.dot(*p) - plane.distance
        }
    }

    fn point_contents(hull: &Hull, num: i16, p: Vec3) -> i16 {
        let mut n = num;
        while n >= 0 {
            let clip_node = &hull.clip_nodes[n as usize];
            let plane = &hull.planes[clip_node.plane_id as usize];

            let d = Self::plane_distance(&plane, &p);

            n = if d < 0.0 {
                clip_node.back
            }
            else {
                clip_node.front
            };
        }

        n
    }

    pub fn get_hull(&self, model_index: i32, hull_index: i32) -> &Hull {
        let model: &Model = &self.models[model_index as usize];
        &model.hulls[hull_index as usize]
    }

    pub fn hull_for_entity(&self, entity: &EntityData, mins: Vec3, maxs: Vec3, offset: Vec3) -> Hull {
        if entity.solid == SolidType::BSP {
            assert!(entity.move_type == MoveType::Push);

            let model: &Model = &self.models[entity.model_index as usize];

            let size = maxs - mins;

            let hull_index = if size.x < 3.0 {
                0
            } else if size.x < 32.0 {
                1
            } else {
                2
            };

            model.hulls[hull_index as usize].clone()
        }
        else {
            unimplemented!();
            
            // let hull_mins = entity.mins - maxs;
            // let hull_maxs = entity.maxs - mins;

            // let hull = SV_HullForBox(hull_mins, hull_maxs);
            //
            // offset = entity.origin;
        }
    }

    pub fn clip_move_to_entity(entity: &EntityData, start: Vec3, mins: Vec3, maxs: Vec3, end: Vec3) {
        let mut hit = HitResult {
            all_solid: true,
            start_solid: false,
            plane: Default::default(),
            fraction: 1.0,
            end_position: end,
        };

        // get hull for entity

    }

    pub fn trace_move(&self, start: Vec3, mins: Vec3, maxs: Vec3, end: Vec3, _type: i32) {
        let hit: HitResult = HitResult::default();

        let (world_entity, _) = self.entities.first().unwrap();
        Self::clip_move_to_entity(world_entity, start, mins, maxs, end);
    }

    pub fn recursive_hull_check(hull: &Hull, num: i32, p1f: f32, p2f: f32, p1: &Vec3, p2: &Vec3, hit: &mut HitResult) -> bool {
        // If we reached a leaf
        if num < 0 {
            if num != ContentsType::Solid as i32 { // CONTENTS_SOLID = -2
                hit.all_solid = false; // at least something is not solid
            }
            else {
                hit.start_solid = true;
            }

            return true;
        }

        // find the distances to the plane
        let clip_node = &hull.clip_nodes[num as usize];
        let plane = &hull.planes[clip_node.plane_id as usize];

        let t1 = Self::plane_distance(plane, &p1);
        let t2 = Self::plane_distance(plane, &p2);

        // if both start and end are on the same side, easy way out
        if t1 >= 0.0 && t2 >= 0.0 {
            return Self::recursive_hull_check(hull, clip_node.front as i32, p1f, p2f, p1, p2, hit);
        }
        else if t1 < 0.0 && t2 < 0.0 {
            return Self::recursive_hull_check(hull, clip_node.back as i32, p1f, p2f, p1, p2, hit);
        }

        // calculate the fraction where we split the line in two
        let epsilon = 0.03125; // 1/32 epsilon to keep floating point happy

        let frac = if t1 < 0.0 {
            ((t1 + epsilon) / (t1 - t2)).clamp(0.0, 1.0)
        } else {
            ((t1 - epsilon) / (t1 - t2)).clamp(0.0, 1.0)
        };

        // calculate the mid point
        let midf = p1f + (p2f - p1f) * frac;
        let mid = p1 + (p2 - p1) * frac;

        // which side of the plane we start the trace from
        let side = if t1 < 0.0 { 1 } else { 0 };
        let children = [ clip_node.front, clip_node.back ];

        // recurse into the near side check
        if !Self::recursive_hull_check(hull, children[side] as i32, p1f, midf, p1, &mid, hit) {
            // if we hit something in our near recurse, trace has its impact point
            return false;
        }

        // if we didn't find a hit yet, check if the other side is not solid, and recurse into it
        if Self::point_contents(hull, children[side^1], mid) != -2 {
            return Self::recursive_hull_check(hull, children[side^1] as i32, midf, p2f, &mid, p2, hit);
        }

        // seems we never got out of solid
        if hit.all_solid {
            return false;
        }

        // so by now we should have our impact point
        hit.plane = if side == 0 {
            Plane {
                normal: plane.normal,
                d: plane.distance,
            }
        }
        else {
            Plane {
                normal: plane.normal * -1.0,
                d: plane.distance,
            }
        };

        hit.fraction = midf;
        hit.end_position = mid;

        false
    }
}