use std::collections::HashMap;
use std::fs::File;
use std::{io, ptr};
use std::io::{Read, Seek, SeekFrom};
use std::sync::Arc;
use byteorder::{LittleEndian, ReadBytesExt};
use glam::Vec3;
use crate::math::{BBoxShort, BoundBox};
use crate::renderer::Texture;

pub enum BSPLumps {
    Entities = 0,
    Planes,
    Textures,
    Vertices,
    Visibility, //vislist
    Nodes,
    TexInfo,
    Faces,
    Lighting,
    ClipNodes,
    Leafs,
    MarkSurfaces, // lfaces
    Edges,
    SurfEdges, // ledges
    Models,
    MaxLumps,
}

#[repr(C)]
pub struct BSPLump {
    pub offset: u32,
    pub size: u32,
}

#[repr(C)]
pub struct BSPHeader {
    pub version: i32,
    pub lumps: [BSPLump; BSPLumps::MaxLumps as usize],
}

#[repr(C)]
pub struct BSPModel {
    pub bound: BoundBox,
    pub origin: Vec3,
    pub head_node: [i32; 4],
    pub numleafs: i32,
    pub face_id: i32,
    pub face_num: i32,
}

#[repr(C)]
#[derive(Debug)]
pub struct BSPNode {
    pub plane_id: i32,
    pub front: u16,
    pub back: u16,
    pub bound: BBoxShort,
    pub face_id: u16,
    pub face_num: u16,
}

#[repr(C)]
pub struct BSPLeaf {
    pub leaf_type: i32,     // type of leaf
    pub vislist: i32,       //
    pub bound: BBoxShort,   // bounding box
    pub lface_id: u16,      // First item of the list of faces. [0,numlfaces]
    pub lface_num: u16,     // Number of faces in the leaf
    pub sndwater: u8,
    pub sndsky: u8,
    pub sndslime: u8,
    pub sndlava: u8,
}

#[repr(C)]
pub struct BSPPlane {
    pub normal: Vec3,
    pub distance: f32,
    pub plane_type: u32,    // 0: Axial plane, in X
                            // 1: Axial plane, in Y
                            // 2: Axial plane, in Z
                            // 3: Non axial plane, roughly toward X
                            // 4: Non axial plane, roughly toward Y
                            // 5: Non axial plane, roughly toward Z
}

#[repr(C)]
pub struct BSPFace {
    pub plane_id: u16,      // the plane in which the face lies. [0,numplanes]
    pub side: u16,          // 0 = front of the plane, 1 = behind of the plane
    pub ledge_id: i32,      // first edge in the list of edges [0,numledges]
    pub ledge_num: u16,     // number of edges in hte list of edges

    pub texinfo_id: u16,    // index of the TextureInfo the face is part of [0,numtexinfos]
    pub typelight: u8,      // type of lighting for the face
    pub baselight: u8,      // 0xFF (dark) to 0x00 (bright)
    pub light: [u8;2],      // additional light models
    pub lightmap: i32,      // pointer inside the general light map or -1
}

#[repr(C)]
pub struct BSPEdge {
    pub vertex0: u16,   // index of the start vertex [0,numvertices]
    pub vertex1: u16,   // index of the end vertex [0,numvertices]
}

#[repr(C)]
pub struct BSPClipNode {
    pub plane_id: u32,
    pub front: i16,     // If positive, id of Front child node
    // If -2, the Front part is inside the model
    // If -1, the Front part is outside the model
    pub back: i16,      // If positive, id of Back child node
    // If -2, the Back part is inside the model
    // If -1, the Back part is outside the model
}

#[repr(C)]
pub struct BSPSurface {
    pub u_axis: Vec3,
    pub u_offset: f32,
    pub v_axis: Vec3,
    pub v_offset: f32,
    pub texture_id: u32,
    animated: u32,
}

#[repr(C)]
#[derive(Debug)]
pub struct BSPTextureHeader             // Mip Texture
{
    pub name: [u8;16],           // Name of the texture.
    pub width: u32,              // width of picture, must be a multiple of 8
    pub height: u32,             // height of picture, must be a multiple of 8
    pub offsets: [u32; 4],       // mip0 (w*h) -> mip1 (1/2) -> mip2 (1/4) -> mip4 (1/8)
}

pub fn read_entities(file: &mut File, header: &BSPHeader, lump_index: BSPLumps) -> io::Result<Vec<HashMap<String, String>>> {
    let lump: &BSPLump = header.lumps.get(lump_index as usize).unwrap();
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

pub fn read_textures(file: &mut File, header: &BSPHeader, lump_index: BSPLumps) -> io::Result<Vec<Texture>> {
    let lump: &BSPLump = header.lumps.get(lump_index as usize).unwrap();

    // read palette
    let palette = Arc::new(read_palette()?);

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

        let mut buffer = [0u8; size_of::<BSPTextureHeader>()];
        file.read_exact(&mut buffer)?;

        let header: BSPTextureHeader = unsafe {
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

pub fn read_lump<T>(file: &mut File, header: &BSPHeader, lump_index: BSPLumps) -> io::Result<Vec<T>> {
    assert!(size_of::<T>() > 0, "Cannot read a zero-sized type");
    assert!(align_of::<T>() <= 8, "Struct alignment too large for direct reading");

    let lump: &BSPLump = header.lumps.get(lump_index as usize).unwrap();
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
