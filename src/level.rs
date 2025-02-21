use std::fs::File;
use std::io;
use std::io::{Read, SeekFrom, Seek};
use std::path::Path;

use glam::{Mat4, Vec3};
use crate::renderer::Renderer;

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

    miptex: BSPLump,
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

struct BSPEdge {
    vertex0: u16,   // index of the start vertex [0,numvertices]
    vertex1: u16,   // index of the end vertex [0,numvertices]
}

pub struct Level {
    bsp_models: Vec<BSPModel>,
    bsp_nodes: Vec<BSPNode>,
    bsp_planes: Vec<BSPPlane>,
    bsp_leafs: Vec<BSPLeaf>,
    bsp_faces: Vec<BSPFace>,
    bsp_lfaces: Vec<u16>,
    bsp_edges: Vec<BSPEdge>,
    bsp_ledges: Vec<i32>,
    vertices: Vec<Vec3>,

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

        // let entities = Self::read_entities(&mut file, &bsp_header.entities).expect("Failed to read entities");

        Ok(Self {
            bsp_models: bsp_models,
            bsp_nodes: bsp_nodes,
            bsp_planes: bsp_planes,
            bsp_leafs: bsp_leafs,
            bsp_faces: bsp_faces,
            bsp_lfaces: bsp_lfaces,
            bsp_edges: bsp_edges,
            bsp_ledges: list_edges,
            vertices: vertices,
        })
    }

    pub fn draw(&self, w: &Mat4, wvp: &Mat4, position: Vec3, renderer: &mut Renderer) {
        // let leaf = self.find_leaf(0, position);

        let mut count = 0;

        self.bsp_leafs.iter().for_each(|leaf| {
            count += 1;
            if count < 500 {
                self.render_leaf(leaf, &w, &wvp, renderer);
            }
        });

        //self.render_leaf(leaf, &w, &wvp, renderer);
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

    fn read_entities(file: &mut File, lump: &BSPLump) -> io::Result<Vec<String>> {
        file.seek(SeekFrom::Start(lump.offset as u64))?;

        let mut buffer = vec![0u8; lump.size as usize];
        file.read_exact(&mut buffer)?;

        let result = String::from_utf8_lossy(&buffer).to_string();

        let lines: Vec<String> = result.lines().map(|s| s.to_string()).collect();

        Ok(lines)
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
        // println!("numfaces {:?}", leaf.lface_num);

        let mut leaf_vertices : Vec<Vec3> = Vec::new();
        let mut leaf_indices : Vec<u32> = Vec::new();

        for face_list_index in leaf.lface_id..(leaf.lface_id + leaf.lface_num) {
            let face_index: u16 = self.bsp_lfaces[face_list_index as usize];
            let face: &BSPFace = &self.bsp_faces[face_index as usize];

            // println!("draw face {:?}", face_index);

            let mut face_vertices : Vec<Vec3> = Vec::new();
            let mut face_indices : Vec<u32> = Vec::new();

            for edge_list_index in face.ledge_id..(face.ledge_id + (face.ledge_num as i32)) {
                let edge_index = self.bsp_ledges[edge_list_index as usize];

                if edge_index >= 0 {
                    let edge = &self.bsp_edges[edge_index as usize];
                    let v0 = self.vertices[edge.vertex0 as usize];
                    let v1 = self.vertices[edge.vertex1 as usize];
                    // println!("v0 {:?} v1 {:?}", v0, v1);

                    face_vertices.push(v0);
                }
                else {
                    let edge = &self.bsp_edges[-edge_index as usize];
                    let v0 = self.vertices[edge.vertex1 as usize];
                    let v1 = self.vertices[edge.vertex0 as usize];
                    // println!("v0 {:?} v1 {:?}", v0, v1);

                    face_vertices.push(v0);
                }
            }

            for i in 2..face_vertices.len() {
                face_indices.push(0);

                face_indices.push((i - 1) as u32);
                face_indices.push(i as u32);
            }

            renderer.draw(&face_vertices, &face_indices, w, wvp);
        }
    }
}

