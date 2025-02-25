use crate::renderer::{Renderer, Texture, Vertex};
use byteorder::{LittleEndian, ReadBytesExt};
use glam::{vec2, Mat4, Vec3};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;
use std::{io, ptr};
use std::collections::HashMap;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct BoundBox {
    min: Vec3, // Minimum X, Y, Z
    max: Vec3, // Maximum X, Y, Z
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct BBoxShort {
    min: [i16;3], // Minimum X, Y, Z
    max: [i16;3], // Maximum X, Y, Z
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
    plane_type: u32,
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
struct Surface {
    vector_s: Vec3,
    dist_s: f32,
    vector_t: Vec3,
    dist_t: f32,
    texture_id: u32,
    animated: u32,
}

#[repr(C)]
#[derive(Debug)]
struct  TextureHeader             // Mip Texture
{
    name: [u8;16],           // Name of the texture.
    width: u32,              // width of picture, must be a multiple of 8
    height: u32,             // height of picture, must be a multiple of 8
    offsets: [u32; 4],
    // offset1: u32,            // offset to u_char Pix[width   * height]
    // offset2: u32,            // offset to u_char Pix[width/2 * height/2]
    // offset4: u32,            // offset to u_char Pix[width/4 * height/4]
    // offset8: u32,            // offset to u_char Pix[width/8 * height/8]
}

#[derive(Debug, Default)]
pub struct Entity {
    properties: HashMap<String, String>,
}

impl Entity {
    pub fn get_property(&self, name: &str) -> Option<&String> {
        self.properties.get(name) // This already returns `Option<&String>`
    }
}

pub struct Level {
    _bsp_models: Vec<BSPModel>,
    bsp_nodes: Vec<BSPNode>,
    bsp_planes: Vec<BSPPlane>,
    bsp_leafs: Vec<BSPLeaf>,
    bsp_faces: Vec<BSPFace>,
    bsp_lfaces: Vec<u16>,
    bsp_edges: Vec<BSPEdge>,
    bsp_ledges: Vec<i32>,
    vislist: Vec<u8>,
    vertices: Vec<Vec3>,
    texture_infos: Vec<Surface>,
    textures: Vec<Texture>,
    entities: Vec<Entity>,
}


impl Level {
    pub(crate) fn load<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let mut file = File::open(path)?;

        // read header
        let mut buffer = vec![0u8; size_of::<BSPHeader>()];
        file.read_exact(&mut buffer)?;
        let bsp_header = unsafe { std::ptr::read(buffer.as_ptr() as *const BSPHeader) };

        if bsp_header.version != 29 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "unsupported bsp version"));
        }

        // read bsp nodes
        let bsp_models: Vec<BSPModel> = Self::read_lump(&mut file, &bsp_header.models).expect("bsp model load failed");
        let bsp_nodes: Vec<BSPNode> = Self::read_lump(&mut file, &bsp_header.nodes).expect("Failed to read BSP nodes");
        let bsp_planes: Vec<BSPPlane> = Self::read_lump(&mut file, &bsp_header.planes).expect("Failed to read BSP planes");
        let bsp_leafs: Vec<BSPLeaf> = Self::read_lump(&mut file, &bsp_header.leaves).expect("Failed to read BSP leafs");
        let bsp_faces: Vec<BSPFace> = Self::read_lump(&mut file, &bsp_header.faces).expect("Failed to read BSP faces");

        let bsp_lfaces: Vec<u16> = Self::read_lump(&mut file, &bsp_header.lface).expect("Failed to read BSP list of faces");
        let bsp_edges: Vec<BSPEdge> = Self::read_lump(&mut file, &bsp_header.edges).expect("Failed to read BSP edges");
        let list_edges: Vec<i32> = Self::read_lump(&mut file, &bsp_header.ledges).expect("Failed to read BSP list of edges");

        let vertices: Vec<Vec3> = Self::read_lump(&mut file, &bsp_header.vertices).expect("Failed to read vertices");

        let vislist: Vec<u8> = Self::read_lump(&mut file, &bsp_header.vislist).expect("Failed to read vislist");

        let texture_infos: Vec<Surface> = Self::read_lump(&mut file, &bsp_header.texinfo).expect("Failed to read texture infos");

        let textures: Vec<Texture> = Self::read_textures(&mut file, &bsp_header.textures).expect("Failed to read textures");

        // let textures: Vec<BSelf::read_lump(&mut file, &bsp_header.textures).expect("Failed to read textures lump");

        let entities = Self::read_entities(&mut file, &bsp_header.entities).expect("Failed to read entities");

        // Self::dump_textures_as_bmp(&textures);

        Ok(Self {
            _bsp_models: bsp_models,
            bsp_nodes: bsp_nodes,
            bsp_planes: bsp_planes,
            bsp_leafs: bsp_leafs,
            bsp_faces: bsp_faces,
            bsp_lfaces: bsp_lfaces,
            bsp_edges: bsp_edges,
            bsp_ledges: list_edges,
            vertices: vertices,
            vislist: vislist,
            texture_infos: texture_infos,
            textures,
            entities,
        })
    }

    pub fn get_entity(&self, name: &str) -> Option<&Entity> {
        for entity in &self.entities {
            if entity.properties.contains_key("classname") && entity.properties["classname"] == name {
                return Some(entity);
            }
        }

        None
    }

    pub fn draw(&self, w: &Mat4, wvp: &Mat4, position: Vec3, renderer: &mut Renderer) {
        // TODO: This potentially can be skipped unless the player moves to another leaf

        let leaf = self.find_leaf(0, position);

        let mut visible_leafs: Vec<usize> = vec![];

        let num_leaves = self.bsp_leafs.len();

        // leaf.vislist marks the offset where the visibility list for this leaf starts
        let mut v = leaf.vislist as usize;

        let mut l = 1;
        while l < num_leaves
        {
            if self.vislist[v] == 0
            {
                // if we read a 0, the next byte tells us how many bytes to skip (RLE)
                // each bit represents a leaf, so we skip that amount of leaf indices (L)
                l += 8 * self.vislist[v + 1] as usize;
                v += 1;
            }
            else
            {
                // tag the 8 leafs in this byte
                for bit in 0..=7 {
                    if self.vislist[v] & (1u8 << bit) != 0 {
                        if l < num_leaves {
                            visible_leafs.push(l);
                        }
                    }
                    l += 1;
                }
            }

            v += 1;
        }

        // self.bsp_leafs.iter().for_each(|leaf| {
        //     self.render_leaf(leaf, &w, &wvp, renderer);
        // });

        visible_leafs.iter().for_each(|l| {
            let leaf = &self.bsp_leafs[*l];
            self.render_leaf(leaf, &w, &wvp, renderer);
        });
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

    fn read_textures(file: &mut File, lump: &BSPLump) -> io::Result<Vec<Texture>> {
        // read palette
        let palette = Arc::new(Self::read_palette()?);

        let mut textures: Vec<Texture> = vec![];

        file.seek(SeekFrom::Start(lump.offset as u64))?;

        let num_textures: i32 = file.read_i32::<LittleEndian>()?;

        let mut offsets = vec![0i32; num_textures as usize];
        file.read_i32_into::<LittleEndian>(&mut offsets)?;

        for offset in offsets {
            if offset == -1 {
                textures.push(Texture {
                    name: "null".to_string(),
                    width: 0,
                    height: 0,
                    data: vec![],
                    palette: palette.clone(),
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

            textures.push(Texture {
                name: name,
                width: header.width,
                height: header.height,
                data: data,
                palette: palette.clone(),
            });
        }

        Ok(textures)
    }

    fn read_entities(file: &mut File, lump: &BSPLump) -> io::Result<Vec<Entity>> {
        file.seek(SeekFrom::Start(lump.offset as u64))?;

        let mut buffer = vec![0u8; lump.size as usize];
        file.read_exact(&mut buffer)?;

        let data = String::from_utf8_lossy(&buffer).to_string();

        let mut entities = Vec::new();
        let mut entity = Entity::default();

        for line in data.lines() {
            let line = line.trim();

            if line == "{" {
                // New entity starts (nothing needed here)
            } else if line == "}" {
                // Push entity and reset
                entities.push(std::mem::take(&mut entity));
            } else if let Some((key, rest)) = line.split_once('"') {
                if let Some((key, rest)) = rest.split_once('"') {
                    if let Some((_, value)) = rest.split_once('"') {
                        if let Some((value, _)) = value.split_once('"') {
                            entity.properties.insert(key.to_string(), value.to_string());
                        }
                    }
                }
            }
        }

        Ok(entities)
    }

    fn find_leaf(&self, node_index: u16, position: Vec3) -> &BSPLeaf {
        let mut n = node_index;
        while n & 0x8000 == 0 {
            let node = &self.bsp_nodes[n as usize];
            let plane = &self.bsp_planes[node.plane_id as usize];

            let distance: f32 = plane.normal.dot(position) - plane.distance;

            if distance >= 0.0 {
                n = node.front
            }
            else {
                n = node.back
            }
        }

        let leaf_index = !n;

        &self.bsp_leafs[leaf_index as usize]
    }

    fn render_leaf(&self, leaf: &BSPLeaf, w: &Mat4, wvp: &Mat4, renderer: &mut Renderer) {
        for face_list_index in leaf.lface_id..(leaf.lface_id + leaf.lface_num) {
            let face_index: u16 = self.bsp_lfaces[face_list_index as usize];
            let face: &BSPFace = &self.bsp_faces[face_index as usize];
            let tex_info: &Surface = &self.texture_infos[face.texinfo_id as usize];
            let texture: &Texture = &self.textures[tex_info.texture_id as usize];

            // let mut face_vertices : Vec<Vec3> = Vec::new();
            let mut face_vertices: Vec<Vertex> = vec![];
            let mut face_indices : Vec<u32> = Vec::new();

            for edge_list_index in face.ledge_id..(face.ledge_id + (face.ledge_num as i32)) {
                let edge_index = self.bsp_ledges[edge_list_index as usize];

                let vertex: Vec3;
                if edge_index >= 0 {
                    let edge = &self.bsp_edges[edge_index as usize];
                    vertex = self.vertices[edge.vertex0 as usize];
                }
                else {
                    let edge = &self.bsp_edges[-edge_index as usize];
                    vertex = self.vertices[edge.vertex1 as usize];
                }

                let uv = vec2(
                    (vertex.dot(tex_info.vector_s) + tex_info.dist_s) / texture.width as f32,
                    (vertex.dot(tex_info.vector_t) + tex_info.dist_t) / texture.height as f32,
                );

                face_vertices.push(Vertex {
                    position: vertex,
                    normal: Vec3::ZERO,
                    tex_coord: uv,
                });
            }

            for i in 2..face_vertices.len() {
                face_indices.push(0);

                face_indices.push(i as u32);
                face_indices.push((i - 1) as u32);
            }

            renderer.draw(&face_vertices, &face_indices, w, wvp, texture);
        }
    }
}

