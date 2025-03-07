use crate::backbuffer::BackBuffer;
use crate::doors::FuncDoor;
use crate::engine::{DebugStats, Engine};
use crate::entity;
use crate::entity::{Entity, EntityData, InfoEntity};
use crate::plane::Plane;
use crate::renderer::{Texture, Vertex};
use byteorder::{LittleEndian, ReadBytesExt};
use glam::{ivec2, vec2, vec4, Mat4, Vec2, Vec3, Vec4};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;
use std::{io, ptr};

enum ModelType {
    Brush,
    Sprite,
    Alias
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct BoundBox {
    pub min: Vec3, // Minimum X, Y, Z
    pub max: Vec3, // Maximum X, Y, Z
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct BBoxShort {
    pub min: [i16;3], // Minimum X, Y, Z
    pub max: [i16;3], // Maximum X, Y, Z
}

#[repr(C)]
struct BSPLump {
    offset: u32,
    size: u32,
}

#[repr(C)]
struct BSPHeader {
    version: i32,

    entities: BSPLump,
    planes: BSPLump, // Map Planes. size/sizeof(plane_t)

    textures: BSPLump,
    vertices: BSPLump,

    vislist: BSPLump,
    nodes: BSPLump,

    texinfo: BSPLump,

    faces: BSPLump,

    lightmaps: BSPLump,
    clipnodes: BSPLump,

    leaves: BSPLump,

    lface: BSPLump,
    edges: BSPLump,

    ledges: BSPLump,
    models: BSPLump,
}

#[repr(C)]
struct BSPModel {
    bound: BoundBox,
    origin: Vec3,
    node_id0: i32,
    node_id1: i32,
    node_id2: i32,
    node_id3: i32,
    numleafs: i32,
    face_id: i32,
    face_num: i32,
}

#[repr(C)]
#[derive(Debug)]
struct BSPNode {
    plane_id: i32,
    front: u16,
    back: u16,
    bound: BBoxShort,
    face_id: u16,
    face_num: u16,
}

#[repr(C)]
struct BSPLeaf {
    leaf_type: i32,     // type of leaf
    vislist: i32,       //
    bound: BBoxShort,   // bounding box
    lface_id: u16,      // First item of the list of faces. [0,numlfaces]
    lface_num: u16,     // Number of faces in the leaf
    sndwater: u8,
    sndsky: u8,
    sndslime: u8,
    sndlava: u8,
}

#[repr(C)]
struct BSPPlane {
    normal: Vec3,
    distance: f32,
    plane_type: u32,    // 0: Axial plane, in X
                        // 1: Axial plane, in Y
                        // 2: Axial plane, in Z
                        // 3: Non axial plane, roughly toward X
                        // 4: Non axial plane, roughly toward Y
                        // 5: Non axial plane, roughly toward Z
}

#[repr(C)]
struct BSPFace {
    plane_id: u16,  // the plane in which the face lies. [0,numplanes]
    side: u16,      // 0 = front of the plane, 1 = behind of the plane
    ledge_id: i32,  // first edge in the list of edges [0,numledges]
    ledge_num: u16, // number of edges in hte list of edges

    texinfo_id: u16,    // index of the TextureInfo the face is part of [0,numtexinfos]
    typelight: u8,      // type of lighting for the face
    baselight: u8,      // 0xFF (dark) to 0x00 (bright)
    light: [u8;2],      // additional light models
    lightmap: i32,      // pointer inside the general light map or -1
}

#[repr(C)]
struct BSPEdge {
    vertex0: u16,   // index of the start vertex [0,numvertices]
    vertex1: u16,   // index of the end vertex [0,numvertices]
}

#[repr(C)]
struct BSPClipNode {
    plane_id: u32,
    front: i16,     // If positive, id of Front child node
                    // If -2, the Front part is inside the model
                    // If -1, the Front part is outside the model
    back: i16,      // If positive, id of Back child node
                    // If -2, the Back part is inside the model
                    // If -1, the Back part is outside the model
}

#[repr(C)]
struct Surface {
    u_axis: Vec3,
    u_offset: f32,
    v_axis: Vec3,
    v_offset: f32,
    texture_id: u32,
    animated: u32,
}

#[repr(C)]
#[derive(Debug)]
struct TextureHeader             // Mip Texture
{
    name: [u8;16],           // Name of the texture.
    width: u32,              // width of picture, must be a multiple of 8
    height: u32,             // height of picture, must be a multiple of 8
    offsets: [u32; 4],       // mip0 (w*h) -> mip1 (1/2) -> mip2 (1/4) -> mip4 (1/8)
}

#[derive(Debug, Default)]
pub struct BSPEntity {
    properties: HashMap<String, String>,
}

impl BSPEntity {
    pub fn get_property(&self, name: &str) -> Option<&String> {
        self.properties.get(name) // This already returns `Option<&String>`
    }
}

#[derive(Debug, Default)]
pub struct HitResult {
    pub all_solid: bool,
    pub start_solid: bool,
    pub plane: Plane,
    pub fraction: f32,
    pub end_position: Vec3,
}

pub struct Level {
    models: Vec<BSPModel>,
    textures: Vec<Texture>,
    lightmaps: Vec<u8>,

    vertices: Vec<Vec3>,
    edges: Vec<BSPEdge>,
    faces: Vec<BSPFace>,
    surfaces: Vec<Surface>,

    planes: Vec<BSPPlane>,
    nodes: Vec<BSPNode>,
    leafs: Vec<BSPLeaf>,
    clip_nodes: Vec<BSPClipNode>,

    vislist: Vec<u8>,
    list_faces: Vec<u16>,
    list_edges: Vec<i32>,

    current_leaf_index: u16,
    visible_leafs: Vec<u16>,

    light_animations: Vec<&'static [u8]>,

    textures_map: HashMap<String, u16>,

    entities: Vec<(EntityData, Box<dyn Entity>)>,

    creators: HashMap<String, Box<dyn Fn() -> Box<dyn Entity>>>,
}


impl Level {
    pub fn load<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let mut file = File::open(path)?;

        // read header
        let mut buffer = vec![0u8; size_of::<BSPHeader>()];
        file.read_exact(&mut buffer)?;
        let bsp_header = unsafe { std::ptr::read(buffer.as_ptr() as *const BSPHeader) };

        if bsp_header.version != 29 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "unsupported bsp version"));
        }

        // read bsp nodes
        let entities_desc = Self::read_entities(&mut file, &bsp_header.entities).expect("Failed to read entities");
        let planes: Vec<BSPPlane> = Self::read_lump(&mut file, &bsp_header.planes).expect("Failed to read BSP planes");
        let textures: Vec<Texture> = Self::read_textures(&mut file, &bsp_header.textures).expect("Failed to read textures");
        let vertices: Vec<Vec3> = Self::read_lump(&mut file, &bsp_header.vertices).expect("Failed to read vertices");
        let vislist: Vec<u8> = Self::read_lump(&mut file, &bsp_header.vislist).expect("Failed to read vislist");
        let nodes: Vec<BSPNode> = Self::read_lump(&mut file, &bsp_header.nodes).expect("Failed to read BSP nodes");
        let texture_infos: Vec<Surface> = Self::read_lump(&mut file, &bsp_header.texinfo).expect("Failed to read texture infos");
        let faces: Vec<BSPFace> = Self::read_lump(&mut file, &bsp_header.faces).expect("Failed to read BSP faces");
        let lightmaps: Vec<u8> = Self::read_lump(&mut file, &bsp_header.lightmaps).expect("Failed to read lightmaps");
        let clip_nodes: Vec<BSPClipNode> = Self::read_lump(&mut file, &bsp_header.clipnodes).expect("Failed to read clip nodes");
        let leafs: Vec<BSPLeaf> = Self::read_lump(&mut file, &bsp_header.leaves).expect("Failed to read BSP leafs");
        let bsp_lfaces: Vec<u16> = Self::read_lump(&mut file, &bsp_header.lface).expect("Failed to read BSP list of faces");
        let bsp_edges: Vec<BSPEdge> = Self::read_lump(&mut file, &bsp_header.edges).expect("Failed to read BSP edges");
        let list_edges: Vec<i32> = Self::read_lump(&mut file, &bsp_header.ledges).expect("Failed to read BSP list of edges");
        let models: Vec<BSPModel> = Self::read_lump(&mut file, &bsp_header.models).expect("bsp model load failed");

        let mut textures_map: HashMap<String, u16> = HashMap::new();
        for i in 0..textures.len() {
            let texture = &textures[i];
            textures_map.insert(texture.name.clone(), i as u16);
        }

        let mut level = Self {
            models,
            nodes,
            planes,
            leafs,
            faces,
            list_faces: bsp_lfaces,
            edges: bsp_edges,
            list_edges,
            vertices,
            vislist,
            surfaces: texture_infos,
            textures,
            lightmaps,
            entities: Vec::new(),
            current_leaf_index: 0,
            visible_leafs: vec![],
            light_animations: vec![
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
            ],
            textures_map,
            clip_nodes,
            creators: HashMap::new(),
        };

        level.register::<FuncDoor>("func_door");

        for entity_desc in entities_desc {
            level.create_entity(&entity_desc);
        }

        Ok(level)
    }

    // fn cast_entity_unchecked<T: 'static>(entity: &dyn Entity) -> &T {
    //     entity.as_any().downcast_ref::<T>().expect("Entity is not expected type")
    // }
    //
    // pub fn get_entity_of_class_name_as<T: 'static>(&self, name: &str) -> Option<&T> {
    //     for entity in &self.entities {
    //         if entity.class_name == name {
    //             return Some(entity);
    //         }
    //     }
    //
    //     None
    // }

    pub fn get_entity_of_class_name(&self, name: &str) -> Option<&EntityData> {
        for (entity, behaviour) in &self.entities {
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
        let classname = &properties["classname"];

        let entity = EntityData {
            class_name: classname.clone(),
            origin: entity::parse_vec3(properties, "origin").unwrap_or(Vec3::ZERO),
            angle: entity::parse_f32(properties, "angle").unwrap_or(0.0),
            model_index: entity::parse_model(properties, "model").unwrap_or(-1),
        };
        let behaviour = self.creators
            .get(classname.as_str())
            .map(|creator| creator());

        if behaviour.is_some() {
            self.entities.push((entity, behaviour.unwrap()));
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

        let start_leaf = &self.leafs[current_leaf_index as usize];

        self.visible_leafs = vec![];
        let num_leaves = self.leafs.len() as u16;

        // leaf.vislist marks the offset where the visibility list for this leaf starts
        let mut v = start_leaf.vislist as usize;

        let mut l: u16 = 1;
        while l < num_leaves
        {
            if self.vislist[v] == 0
            {
                // if we read a 0, the next byte tells us how many bytes to skip (RLE)
                // each bit represents a leaf, so we skip that amount of leaf indices (L)
                l += 8 * self.vislist[v + 1] as u16;
                v += 1;
            }
            else
            {
                // tag the 8 leafs in this byte
                for bit in 0..=7 {
                    if self.vislist[v] & (1u8 << bit) != 0 {
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
        let node = &self.nodes[0];
        let mut visible_leafs: Vec<u16> = vec![];
        self.traverse_bsp_tree(engine, stats, back_buffer, node, position, &frustum_planes, w, wvp, &mut drawn_faces, &mut visible_leafs);

        stats.leafs_visible += visible_leafs.len() as u32;

        // Now collect faces, faces can be repeated in different leafs (AFAIK)
        let mut visible_faces: Vec<u16> = vec![];
        for leaf_index in visible_leafs {
            // self.draw_leaf(engine, stats, back_buffer, leaf, drawn_faces, w, wvp);
            let leaf: &BSPLeaf = &self.leafs[leaf_index as usize];

            let face_list_offset = leaf.lface_id;
            let face_list_num = face_list_offset + leaf.lface_num;

            for face_list_index in face_list_offset..face_list_num {
                let face_index: u16 = self.list_faces[face_list_index as usize];
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
        for (entity, behaviour) in &self.entities
        {
            let model_index = entity.model_index;
            if model_index >= 1 {
                let model: &BSPModel = &self.models[model_index as usize];
                self.draw_model(model, engine, back_buffer, &frustum_planes, w, wvp, stats);
            }
        }
    }

    fn draw_model(&self, model: &BSPModel, engine: &Engine, back_buffer: &mut BackBuffer, frustum_planes: &[Plane;6], w: &Mat4, wvp: &Mat4, stats: &mut DebugStats) {
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

        let plane = &self.planes[node.plane_id as usize];
        let distance: f32 = plane.normal.dot(position) - plane.distance;

        let children = if distance >= 0.0 {
            [node.front, node.back]
        }
        else {
            [node.back, node.front]
        };

        for child in children {
            if child & 0x8000 == 0 {
                let first_node = &self.nodes[child as usize];
                self.traverse_bsp_tree(engine, stats, back_buffer, first_node, position, frustum_planes, w, wvp, drawn_faces, visited_leafs);
            } else {
                let leaf_index = !child;

                if self.visible_leafs.contains(&leaf_index) {
                    let leaf = &self.leafs[leaf_index as usize];

                    if Self::is_aabb_outside_frustum_short(frustum_planes, &leaf.bound) {
                        continue;
                    }

                    visited_leafs.push(leaf_index);
                }
            }
        }
    }

    fn read_lump<T>(file: &mut File, lump: &BSPLump) -> io::Result<Vec<T>> {
        assert!(size_of::<T>() > 0, "Cannot read a zero-sized type");
        assert!(align_of::<T>() <= 8, "Struct alignment too large for direct reading");

        file.seek(SeekFrom::Start(lump.offset as u64))?;

        let mut buffer = vec![0u8; lump.size as usize];
        file.read_exact(&mut buffer)?;

        let struct_size = size_of::<T>();
        let count = lump.size as usize / struct_size;

        let structs: Vec<T> = unsafe {
            let ptr = buffer.as_ptr() as *mut T;
            Vec::from_raw_parts(ptr, count, count)
        };

        std::mem::forget(buffer);

        Ok(structs)
    }

    fn read_palette() -> io::Result<[u32;256]> {
        let mut file = File::open("data/palette.lmp")?;
        let mut buffer = vec![0u8; 256 * 3];
        file.read_exact(&mut buffer)?;

        let mut palette = [0u32; 256];
        for i in 0..256 {
            let r = buffer[i * 3] as u32;
            let g = buffer[i * 3 + 1] as u32;
            let b = buffer[i * 3 + 2] as u32;
            let a = 255;
            palette[i] = (a << 24) | (r << 16) | (g << 8) | b;
        }

        Ok(palette)
    }

    fn get_texture(&self, name: &String) -> Option<&Texture> {
        self.textures_map.get(name).and_then(|&index| self.textures.get(index as usize))
    }

    fn read_textures(file: &mut File, lump: &BSPLump) -> io::Result<Vec<Texture>> {
        // read palette
        let palette = Arc::new(Self::read_palette()?);

        let mut textures: Vec<Texture> = vec![];

        file.seek(SeekFrom::Start(lump.offset as u64))?;

        let num_textures: i32 = file.read_i32::<LittleEndian>()?;

        let mut offsets = vec![0i32; num_textures as usize];
        file.read_i32_into::<LittleEndian>(&mut offsets)?;

        let mut map: HashMap<String, u32> = HashMap::new();

        for offset in offsets {
            if offset == -1 {
                textures.push(Texture {
                    name: "null".to_string(),
                    width: 0,
                    height: 0,
                    data: vec![],
                    palette: palette.clone(),
                    frames: 0,
                });

                continue;
            }

            let texture_header_offset = lump.offset as u64 + offset as u64;
            file.seek(SeekFrom::Start(texture_header_offset))?;

            let mut buffer = [0u8; size_of::<TextureHeader>()];
            file.read_exact(&mut buffer)?;

            let header: TextureHeader = unsafe {
                ptr::read_unaligned(buffer.as_ptr() as *const _)
            };

            let end = header.name.iter().position(|&b| b == 0).unwrap_or(header.name.len());
            let name = std::str::from_utf8(&header.name[..end])
                .unwrap_or("Invalid UTF-8")
                .to_string(); // Convert to owned String

            let mut data = vec![];
            for i in 0..4 {
                let mip_data_offset = texture_header_offset + header.offsets[i] as u64;
                file.seek(SeekFrom::Start(mip_data_offset))?;
                let w = header.width >> i;
                let h = header.height >> i;

                let mut buffer = vec![0u8; (w * h) as usize];
                file.read_exact(&mut buffer)?;

                data.push(buffer);
            }

            println!("Loaded texture: {:?}", name);

            // keep track of animated frames count
            if name.starts_with('+') {
                if name.chars().nth(1).unwrap().is_ascii_digit() {
                    let actual_name = &name[2..];
                    let counter = map.entry(actual_name.to_string()).or_insert(0);
                    *counter += 1;
                }
            }

            textures.push(Texture {
                name,
                width: header.width,
                height: header.height,
                data,
                palette: palette.clone(),
                frames: 0,
            });
        }

        for texture in &mut textures {
            if texture.name.starts_with('+') {
                let actual_name = &texture.name[2..];
                if let Some(frames) = map.get(actual_name) {
                    texture.frames = *frames as u8;
                }
            }
        }

        Ok(textures)
    }

    fn read_entities(file: &mut File, lump: &BSPLump) -> io::Result<Vec<HashMap<String, String>>> {
        file.seek(SeekFrom::Start(lump.offset as u64))?;

        let mut buffer = vec![0u8; lump.size as usize];
        file.read_exact(&mut buffer)?;

        let data = String::from_utf8_lossy(&buffer).to_string();

        let mut entities: Vec<HashMap<String, String>> = vec![];
        let mut entity_desc: HashMap<String, String> = HashMap::new();

        for line in data.lines() {
            let line = line.trim();

            if line == "}" {
                entities.push(entity_desc.clone());
                entity_desc.clear();
            } else if let Some((_, rest)) = line.split_once('"') {
                if let Some((key, rest)) = rest.split_once('"') {
                    if let Some((_, value)) = rest.split_once('"') {
                        if let Some((value, _)) = value.split_once('"') {
                            entity_desc.insert(key.to_string(), value.to_string());
                        }
                    }
                }
            }
        }

        Ok(entities)
    }

    fn find_leaf(&self, node_index: u16, position: Vec3) -> u16 {
        let mut n = node_index;
        while n & 0x8000 == 0 {
            let node = &self.nodes[n as usize];
            let plane = &self.planes[node.plane_id as usize];

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

        let tex_info: &Surface = &self.surfaces[face.texinfo_id as usize];

        for edge_list_index in face.ledge_id..(face.ledge_id + (face.ledge_num as i32)) {
            let edge_index = self.list_edges[edge_list_index as usize];

            let vertex: Vec3;
            if edge_index >= 0 {
                let edge = &self.edges[edge_index as usize];
                vertex = self.vertices[edge.vertex0 as usize];
            } else {
                let edge = &self.edges[-edge_index as usize];
                vertex = self.vertices[edge.vertex1 as usize];
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
            let face: &BSPFace = &self.faces[*face_index as usize];
            let tex_info: &Surface = &self.surfaces[face.texinfo_id as usize];
            let mut texture: &Texture = &self.textures[tex_info.texture_id as usize];

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
                let edge_index = self.list_edges[edge_list_index as usize];

                let vertex: Vec3;
                if edge_index >= 0 {
                    let edge = &self.edges[edge_index as usize];
                    vertex = self.vertices[edge.vertex0 as usize];
                }
                else {
                    let edge = &self.edges[-edge_index as usize];
                    vertex = self.vertices[edge.vertex1 as usize];
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
                let lightmap_data: &[u8] = &self.lightmaps[face.lightmap as usize..(face.lightmap + lightmap_size.x * lightmap_size.y) as usize];
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
        let frames = self.light_animations[light_type as usize];

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

    // #define	CONTENTS_EMPTY		-1
    // #define	CONTENTS_SOLID		-2
    // #define	CONTENTS_WATER		-3
    // #define	CONTENTS_SLIME		-4
    // #define	CONTENTS_LAVA		-5
    // #define	CONTENTS_SKY		-6
    // #define	CONTENTS_ORIGIN		-7		// removed at csg time
    // #define	CONTENTS_CLIP		-8		// changed to contents_solid

    fn plane_distance(plane: &BSPPlane, p: &Vec3) -> f32 {
        if plane.plane_type < 3 {
            p[plane.plane_type as usize] - plane.distance
        }
        else {
            plane.normal.dot(*p) - plane.distance
        }
    }

    fn point_contents(&self, num: i16, p: Vec3) -> i16 {
        let mut n = num;
        while n >= 0 {
            let clip_node = &self.clip_nodes[n as usize];
            let plane = &self.planes[clip_node.plane_id as usize];

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

    pub fn trace(&self, p1: Vec3, p2: Vec3, hit: &mut HitResult) -> bool {
        self.trace_segment(0, 0.0, 1.0, &p1, &p2, hit)
    }

    fn trace_segment(&self, num: i16, p1f: f32, p2f: f32, p1: &Vec3, p2: &Vec3, hit: &mut HitResult) -> bool {
        // If we reached a leaf
        if num < 0 {
            if num != -2 { // CONTENTS_SOLID = -2
                hit.all_solid = false; // at least something is not solid
            }
            else {
                hit.start_solid = true;
            }

            return true;
        }

        // find the distances to the plane
        let clip_node = &self.clip_nodes[num as usize];
        let plane = &self.planes[clip_node.plane_id as usize];

        let t1 = Self::plane_distance(plane, &p1);
        let t2 = Self::plane_distance(plane, &p2);

        // if both start and end are on the same side, easy way out
        if t1 >= 0.0 && t2 >= 0.0 {
            return self.trace_segment(clip_node.front, p1f, p2f, p1, p2, hit);
        }
        else if t1 < 0.0 && t2 < 0.0 {
            return self.trace_segment(clip_node.back, p1f, p2f, p1, p2, hit);
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
        if !self.trace_segment(children[side], p1f, midf, p1, &mid, hit) {
            // if we hit something in our near recurse, trace has its impact point
            return false;
        }

        // if we didn't find a hit yet, check if the other side is not solid, and recurse into it
        if self.point_contents(children[side^1], mid) != -2 {
            return self.trace_segment(children[side^1], midf, p2f, &mid, p2, hit);
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
