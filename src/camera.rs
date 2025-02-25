use glam::{Mat4, Quat, Vec3, Vec4};

pub struct Camera {
    pub position: Vec3,
    pub pitch: f32,
    pub yaw: f32,
    pub aspect: f32,
    pub fov: f32,
    pub znear: f32,
    pub zfar: f32,
}

impl Camera {
    pub fn new() -> Camera {
        Camera {
            position: Vec3::ZERO,
            pitch: 0.0,
            yaw: 0.0,
            aspect: 0.0,
            fov: 70.0,
            znear: 1.0,
            zfar: 10000.0,
        }
    }

    pub fn forward(&self) -> Vec3 {
        self.get_world().x_axis.truncate()
    }

    pub fn left(&self) -> Vec3 {
        self.get_world().y_axis.truncate()
    }

    pub fn up(&self) -> Vec3 {
        self.get_world().z_axis.truncate()
    }

    pub fn get_world(&self) -> Mat4 {
        let rotation = Quat::from_euler(glam::EulerRot::ZXY, self.yaw, 0.0, self.pitch);
        Mat4::from_scale_rotation_translation(Vec3::ONE, rotation, self.position)
    }

    pub fn get_view_mat(&self) -> Mat4 {
        let world = self.get_world();
        let view = world.inverse();

        let coordinate = Mat4::from_cols(
            // Column 0: mapping of world X (forward) to view space (0, 0, -1)
            Vec4::new(0.0, 0.0, 1.0, 0.0),
            // Column 1: mapping of world Y (left) to view space (-1, 0, 0)
            Vec4::new(-1.0, 0.0, 0.0, 0.0),
            // Column 2: mapping of world Z (up) to view space (0, 1, 0)
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            // Column 3: translation (unchanged)
            Vec4::new(0.0, 0.0, 0.0, 1.0),
        );

        coordinate * view
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