use glam::{vec3, IVec2, Mat4, Vec3, Vec4};

#[allow(dead_code)]
pub struct IRect {
    pub min: IVec2,
    pub max: IVec2,
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

#[derive(Debug, Default)]
pub struct Plane {
    pub normal: Vec3,
    pub d: f32,
}

impl Plane {
    pub fn new(v: Vec4) -> Plane {
        let normal = v.truncate();               // Take the xyz
        let length = normal.length();             // Normal length
        Plane {
            normal: normal / length,              // Normalize normal
            d: v.w / length                       // Scale d to match
        }
    }
    
    pub fn extract_frustum_planes_world_space(proj: &Mat4, view: &Mat4) -> [Plane; 6] {
        let mut planes = [Vec4::default(); 6];

        let combo = *proj * *view;

        planes[0] = combo.row(3) + combo.row(1); // Left (Y)
        planes[1] = combo.row(3) - combo.row(1); // Right (Y)
        planes[2] = combo.row(3) + combo.row(2); // Bottom (Z)
        planes[3] = combo.row(3) - combo.row(2); // Top (Z)
        planes[4] = combo.row(3) + combo.row(0); // Near (X)
        planes[5] = combo.row(3) - combo.row(0); // Far (X)

        planes.map(Plane::new)
    }

    pub fn is_bboxshort_outside_plane(plane: &Plane, bbox: &BBoxShort) -> bool {
        let corners = [
            [ bbox.min[0], bbox.min[1], bbox.min[2] ],
            [ bbox.min[0], bbox.min[1], bbox.max[2] ],
            [ bbox.min[0], bbox.max[1], bbox.min[2] ],
            [ bbox.min[0], bbox.max[1], bbox.max[2] ],
            [ bbox.max[0], bbox.min[1], bbox.min[2] ],
            [ bbox.max[0], bbox.min[1], bbox.max[2] ],
            [ bbox.max[0], bbox.max[1], bbox.min[2] ],
            [ bbox.max[0], bbox.max[1], bbox.max[2] ],
        ];

        fn distance_to_plane(p: &[i16; 3], plane: &Plane) -> f32 {
            (plane.normal.x * p[0] as f32) +
                (plane.normal.y * p[1] as f32) +
                (plane.normal.z * p[2] as f32) +
                plane.d
        }

        let mut outside_count = 0;
        for corner in corners {
            let d = distance_to_plane(&corner, &plane);
            if d < 0.0 {
                outside_count += 1;
            }
        }

        outside_count == 8
    }

    pub fn is_bbox_outside_plane(plane: &Plane, bbox: &BoundBox) -> bool {
        let corners = [
            vec3(bbox.min.x, bbox.min.y, bbox.min.z),
            vec3(bbox.min.x, bbox.min.y, bbox.max.z),
            vec3(bbox.min.x, bbox.max.y, bbox.min.z),
            vec3(bbox.min.x, bbox.max.y, bbox.max.z),
            vec3(bbox.max.x, bbox.min.y, bbox.min.z),
            vec3(bbox.max.x, bbox.min.y, bbox.max.z),
            vec3(bbox.max.x, bbox.max.y, bbox.min.z),
            vec3(bbox.max.x, bbox.max.y, bbox.max.z),
        ];

        fn distance_to_plane(p: &Vec3, plane: &Plane) -> f32 {
            (plane.normal.x * p.x) + (plane.normal.y * p.y) + (plane.normal.z * p.z) + plane.d
        }

        let mut outside_count = 0;
        for corner in corners {
            let d = distance_to_plane(&corner, &plane);
            if d < 0.0 {
                outside_count += 1;
            }
        }

        outside_count == 8
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Color {
    rgba: Vec4,
}

#[allow(dead_code)]
impl Color {
    pub fn get_vec(&self) -> Vec4 { self.rgba }

    pub fn from_vec4(rgba: Vec4) -> Self { Color { rgba } }

    pub fn from_f32(r: f32, g: f32, b: f32, a: f32) -> Color {
        Self {
            rgba: Vec4::new(r, g, b, a)
        }
    }

    pub fn from_u8(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self {
            rgba: Vec4::new(
                r as f32 / 255.0,
                g as f32 / 255.0,
                b as f32 / 255.0,
                a as f32 / 255.0,
            ),
        }
    }

    pub fn from_u32(rgba: u32) -> Self {
        let a = ((rgba >> 24) & 0xFF) as u8;
        let r = ((rgba >> 16) & 0xFF) as u8;
        let g = ((rgba >> 8) & 0xFF) as u8;
        let b = (rgba & 0xFF) as u8;
        Self::from_u8(r, g, b, a)
    }

    pub fn to_u32(&self) -> u32 {
        let scaled = self.rgba * 255.0;
        ((scaled.w as u32) << 24) | ((scaled.x as u32) << 16) | ((scaled.y as u32) << 8) | scaled.z as u32
    }
}
