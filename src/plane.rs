use glam::{Mat4, Vec3, Vec4};
use crate::level::BBoxShort;

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

    pub fn is_bbox_outside_plane(plane: &Plane, bbox: &BBoxShort) -> bool {
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
}