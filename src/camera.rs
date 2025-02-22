use glam::{Mat4, Vec3, Vec4};

pub struct Camera {
    pub position: Vec3,
    pub aspect: f32,
    pub fov: f32,
    pub znear: f32,
    pub zfar: f32,
}

impl Camera {
    pub fn new() -> Camera {
        Camera {
            position: Vec3::ZERO,
            aspect: 0.0,
            fov: 70.0,
            znear: 1.0,
            zfar: 10000.0
        }
    }

    pub fn get_view_mat(&self) -> Mat4 {
        Mat4::look_to_rh(
            self.position,
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0) // Z is up
        )
    }

    pub fn get_projection_mat(&self) -> Mat4 {
        let (sin_fov, cos_fov) = f32::sin_cos(0.5 * self.fov.to_radians());
        let h = cos_fov / sin_fov;
        let w = h / self.aspect;
        let r = self.zfar / (self.zfar - self.znear);

        Mat4::from_cols(
            Vec4::new(w, 0.0, 0.0, 0.0),
            Vec4::new(0.0, h, 0.0, 0.0),
            Vec4::new(0.0, 0.0, r, 1.0),
            Vec4::new(0.0, 0.0, -r * self.znear, 0.0),
        )
    }
}