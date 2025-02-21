mod model;
mod level;
mod camera;
mod color;
mod renderer;

use crate::level::Level;
use crate::renderer::Renderer;
use camera::Camera;
use glam::{Mat4, Quat, Vec3};
use minifb::{Key, Window, WindowOptions};
use model::Model;

const SCREEN_WIDTH: usize = 1920;
const SCREEN_HEIGHT: usize = 1080;



fn main() {
    // create a window and buffer
    let mut window = Window::new(
        "RustedQuake",
        SCREEN_WIDTH,
        SCREEN_HEIGHT,
        WindowOptions::default()
    ).unwrap_or_else(|e| {
        panic!("{}", e);
    });

    // create our renderer
    let mut renderer = Renderer::new(SCREEN_WIDTH, SCREEN_HEIGHT);

    // load our test obj file
    let model = Model::load("data/cube.obj").expect("Failed to load model");
    let level = Level::load("data/start.bsp").expect("Failed to load level");

    let angles = [0.0,90.0,180.0,270.0];
    let mut x_input = 0;
    let mut y_input = 0;
    let mut z_input = 0;

    let mut camera_angles: Vec3 = Vec3::new(0.0, 0.0, 0.0);

    // create our camera
    let mut camera = Camera::new();
    // camera.position = Vec3::new(-2282.5986, -939.2992, 4787.5225);
    // camera.position = Vec3::new(480.0, 352.0, 88.0);

    // camera.position = Vec3::new(-269.10007, 2353.6995, 597.60004);
    camera.position = Vec3::new(0.0,0.0,0.0);

    camera.aspect = SCREEN_WIDTH as f32 / SCREEN_HEIGHT as f32;

    let camera_speed = 0.1;

    // run the main-loop
    while window.is_open() && !window.is_key_down(Key::Escape) {

        println!("Camera position: {:?}", camera.position);
        // clear our buffer to start rendering
        renderer.clear();

        // angle += 0.01;

        if window.is_key_down(Key::D) {
            camera.position.y += camera_speed;
        }
        if window.is_key_down(Key::A) {
            camera.position.y -= camera_speed;
        }
        if window.is_key_down(Key::E) {
            camera.position.z += camera_speed;
        }
        if window.is_key_down(Key::Q) {
            camera.position.z -= camera_speed;
        }
        if window.is_key_down(Key::S) {
            camera.position.x -= camera_speed;
        }
        if window.is_key_down(Key::W) {
            camera.position.x += camera_speed;
        }
        if window.is_key_released(Key::Key1) {
            x_input += 1;
            if x_input > 3 {
                x_input = 0;
            }

            camera_angles.x = angles[x_input];
        }
        if window.is_key_released(Key::Key2) {
            y_input += 1;
            if y_input > 3 {
                y_input = 0;
            }

            camera_angles.y = angles[y_input];
        }
        if window.is_key_released(Key::Key3) {
            z_input += 1;
            if z_input > 3 {
                z_input = 0;
            }

            camera_angles.z = angles[z_input];
        }

        let view = camera.get_view_mat();
        let proj = camera.get_projection_mat();

        // build our world transform
        let world = Mat4::from_scale_rotation_translation(
            Vec3::new(1.0, 1.0, 1.0),
            Quat::IDENTITY,
            Vec3::new(0.0, 0.0, 0.0)
        );

        let wvp = proj * view * world;

        renderer.draw(&model.vertices, &model.indices, &world, &wvp);

        // level.draw(&world, &wvp, Vec3::new(480.0, 352.0, 88.0), &mut renderer);

        window
            .update_with_buffer(&renderer.get_back_buffer(), SCREEN_WIDTH, SCREEN_HEIGHT)
            .unwrap();
    }
}
