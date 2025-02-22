mod model;
mod level;
mod camera;
mod color;
mod renderer;

use std::time::Instant;
use crate::level::Level;
use crate::renderer::Renderer;
use camera::Camera;
use glam::{Mat4, Quat, Vec3};
use minifb::{Key, Scale, Window, WindowOptions};
use minifb_fonts::{font5x8, font6x8};
use model::Model;
use crate::color::Color;

const SCREEN_WIDTH: usize = 1920;
const SCREEN_HEIGHT: usize = 1080;


fn float_to_u32_color(value: f32) -> u32 {
    let intensity = (value.clamp(0.0, 1.0) * 255.0).round() as u32;
    0xFF000000 | (intensity << 16) | (intensity << 8) | intensity
}

fn main() {
    // create a text renderer
    let text_color: Color = Color::from_u8(255,255,255,255);
    let text = font6x8::new_renderer(SCREEN_WIDTH, SCREEN_HEIGHT, text_color.to_u32());

    // create a window and buffer
    let mut window = Window::new(
        "RustedQuake",
        SCREEN_WIDTH,
        SCREEN_HEIGHT,
        WindowOptions {
            // scale: Scale::X2,
            ..WindowOptions::default()
        }
    ).unwrap_or_else(|e| {
        panic!("{}", e);
    });

    // create our renderer
    let mut renderer = Renderer::new(SCREEN_WIDTH, SCREEN_HEIGHT);

    // load our test obj file
    let model = Model::load("data/teapot.obj").expect("Failed to load model");
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
    camera.position = Vec3::new(-538.0,1000.0,0.0);

    camera.aspect = SCREEN_WIDTH as f32 / SCREEN_HEIGHT as f32;

    let mut camera_speed = 0.1;
    let mut draw_depth = false;

    let mut instant = Instant::now();

    // run the main-loop
    while window.is_open() && !window.is_key_down(Key::Escape) {
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
        if window.is_key_released(Key::Tab) {
            draw_depth = !draw_depth;
        }

        // control camera speed
        if window.is_key_released(Key::NumPadMinus) {
            camera_speed -= 10.0;
        }
        else if window.is_key_released(Key::NumPadPlus) {
            camera_speed += 10.0;
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

        level.draw(&world, &wvp, Vec3::new(480.0, 352.0, 88.0), &mut renderer);

        let mut buffer: Vec<u32>;

        if draw_depth {
            let depth: &[f32] = renderer.get_depth_buffer();
            buffer = depth.iter()
                .map(|&x| (float_to_u32_color(x)))
                .collect();
        }
        else {
            buffer = Vec::from(renderer.get_back_buffer());
        }

        // fps counter
        let elapsed = instant.elapsed().as_secs_f32();
        instant = Instant::now();

        // draw some informative text
        let text_frame_time = format!("frame: {}ms", (elapsed * 1000.0) as u32);
        text.draw_text(&mut buffer, 10, 20, text_frame_time.as_str());
        let text_camera_position = format!("camera: {}", camera.position);
        text.draw_text(&mut buffer, 10, 40, text_camera_position.as_str());

        let stats = renderer.get_stats();
        let str_stats = format!("{:#?}", stats) // Pretty-prints with new lines
            .trim_start_matches("RendererStats {\n") // Remove struct name
            .trim_end_matches("\n}") // Remove closing brace
            .replace("    ", ""); // Remove excess indentation
        text.draw_text(&mut buffer, 10, 80, str_stats.as_str());

        window
            .update_with_buffer(buffer.as_slice(), SCREEN_WIDTH, SCREEN_HEIGHT)
            .unwrap();
    }
}
