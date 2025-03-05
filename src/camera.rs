use glam::{vec2, vec3, Mat4, Quat, Vec2, Vec3, Vec4};
use crate::engine::Engine;
use crate::level::{HitResult, Level};

#[derive(Default)]
pub struct CameraSettings {
    pub aspect: f32,
    pub fov: f32,
    pub znear: f32,
    pub zfar: f32,
}

#[derive(Default)]
pub struct Camera {
    pub camera_settings: CameraSettings,
    pub position: Vec3,
    pub pitch: f32,
    pub yaw: f32,

    pub speed: f32,
    pub rotation_speed: f32,

    movement: Vec3,
    look: Vec2,
}

impl Camera {
    pub fn new(camera_settings: CameraSettings) -> Self {
        Self {
            camera_settings,
            position: Vec3::ZERO,
            pitch: 0.0,
            yaw: 0.0,
            speed: 200.0,
            rotation_speed: 200.0,
            movement: vec3(0.0, 0.0, 0.0),
            look: vec2(0.0, 0.0),
        }
    }

    pub fn set_position(&mut self, position: Vec3) {
        self.position = position;
    }

    pub fn set_rotation(&mut self, yaw: f32, pitch: f32) {
        self.yaw = yaw;
        self.pitch = pitch;
    }

    pub fn set_input(&mut self, movement: Vec3, look: Vec2) {
        self.movement = movement;
        self.look = look;
    }

    pub fn update(&mut self, engine: &Engine, level: &Level) {
        let time = engine.time().elapsed_time();

        // end_position += self.forward() * self.movement.x * self.speed * time;
        // end_position += self.left() * self.movement.y * self.speed * time;
        // end_position += self.up() * self.movement.z * self.speed * time;

        let mut movement = vec3(0.0, 0.0, 0.0);
        movement += self.forward() * self.movement.x * self.speed * time;
        movement += self.left() * self.movement.y * self.speed * time;
        movement += self.up() * self.movement.z * self.speed * time;

        self.yaw += self.look.x * self.rotation_speed * time;
        self.pitch += self.look.y * self.rotation_speed * time;

        // let hit = level.collide(&end_position);
        // if hit == -1 {
        //     self.position = end_position;
        // }

        let mut hit: HitResult = HitResult::default();
        while !level.trace(self.position, self.position + movement, &mut hit) {
            // adjust to slide against wall
            let d = movement.dot(hit.plane.normal);
            movement = movement - hit.plane.normal * d;

            // check again
            // if !level.trace(0, 0.0, 1.0, self.position, self.position + movement, &mut hit) {
            //     movement = Vec3::ZERO;
            // }
        }

        self.position = self.position + movement;
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
        let rotation = Quat::from_euler(glam::EulerRot::ZXY, self.yaw.to_radians(), 0.0, self.pitch.to_radians());
        Mat4::from_scale_rotation_translation(Vec3::ONE, rotation, self.position)
    }

    pub fn get_view_mat(&self) -> Mat4 {
        let world = self.get_world();
        let view = world.inverse();

        let coordinate = Mat4::from_cols(
            Vec4::new(0.0, 0.0, 1.0, 0.0),
            Vec4::new(-1.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 1.0),
        );

        coordinate * view
    }

    pub fn get_projection_mat(&self) -> Mat4 {
        let (sin_fov, cos_fov) = f32::sin_cos(0.5 * self.camera_settings.fov.to_radians());
        let h = cos_fov / sin_fov;
        let w = h / self.camera_settings.aspect;
        let r = self.camera_settings.zfar / (self.camera_settings.zfar - self.camera_settings.znear);

        Mat4::from_cols(
            Vec4::new(w, 0.0, 0.0, 0.0),
            Vec4::new(0.0, h, 0.0, 0.0),
            Vec4::new(0.0, 0.0, r, 1.0),
            Vec4::new(0.0, 0.0, -r * self.camera_settings.znear, 0.0),
        )
    }
}