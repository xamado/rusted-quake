mod model;
mod level;
mod camera;
mod color;
mod renderer;

use crate::color::Color;
use crate::level::Level;
use crate::renderer::{Renderer, RendererSettings};
use camera::Camera;
use glam::{vec3, Mat4, Quat, Vec3};
use minifb::{Key, Scale, Window, WindowOptions};
use minifb_fonts::font6x8;
use model::Model;
use std::time::Instant;

const SCREEN_WIDTH: usize = 1920/2;
const SCREEN_HEIGHT: usize = 1080/2;

struct GameSettings {
    pub camera_speed: f32,
    pub camera_rotation_speed: f32,
    pub draw_depth: bool,
    pub show_stats: bool,
}

fn float_to_u32_color(value: f32) -> u32 {
    let intensity = (value.clamp(0.0, 1.0) * 255.0).round() as u32;
    0xFF000000 | (intensity << 16) | (intensity << 8) | intensity
}

fn process_input(window: &Window, elapsed_seconds: f32, game_settings: &mut GameSettings, camera: &mut Camera) {
    if window.is_key_down(Key::W) {
        camera.position += camera.forward() * game_settings.camera_speed * elapsed_seconds;
    }
    if window.is_key_down(Key::S) {
        camera.position -= camera.forward() * game_settings.camera_speed * elapsed_seconds;
    }
    if window.is_key_down(Key::A) {
        camera.position += camera.left() * game_settings.camera_speed * elapsed_seconds;
    }
    if window.is_key_down(Key::D) {
        camera.position -= camera.left() * game_settings.camera_speed * elapsed_seconds;
    }
    if window.is_key_down(Key::E) {
        camera.position += vec3(0.0, 0.0, 1.0) * game_settings.camera_speed * elapsed_seconds;
    }
    if window.is_key_down(Key::Q) {
        camera.position -= vec3(0.0, 0.0, 1.0) * game_settings.camera_speed * elapsed_seconds;
    }
    if window.is_key_down(Key::Left) {
        camera.yaw += game_settings.camera_rotation_speed * elapsed_seconds;
    }
    if window.is_key_down(Key::Right) {
        camera.yaw -= game_settings.camera_rotation_speed * elapsed_seconds;
    }
    if window.is_key_down(Key::Up) {
        camera.pitch -= game_settings.camera_rotation_speed * elapsed_seconds;
    }
    if window.is_key_down(Key::Down) {
        camera.pitch += game_settings.camera_rotation_speed * elapsed_seconds;
    }
    if window.is_key_released(Key::Tab) {
        game_settings.draw_depth = !game_settings.draw_depth;
    }

    // control camera speed
    if window.is_key_released(Key::NumPadMinus) {
        game_settings.camera_speed -= 10.0;
    }
    else if window.is_key_released(Key::NumPadPlus) {
        game_settings.camera_speed += 10.0;
    }
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
            scale: Scale::X2,
            ..WindowOptions::default()
        }
    ).unwrap_or_else(|e| {
        panic!("{}", e);
    });

    // create our renderer
    let renderer_settings = RendererSettings {
        naive_rasterization: true,
        tile_size: 16,
        wireframe: false,
        ..RendererSettings::default()
    };
    let mut renderer = Renderer::new(SCREEN_WIDTH, SCREEN_HEIGHT, renderer_settings);

    // load our test obj file
    let model = Model::load("data/plane.obj").expect("Failed to load model");
    let level = Level::load("data/e1m2.bsp").expect("Failed to load level");

    let entity_player_start = level.get_entity("info_player_start");

    let player_spawn: Vec3 = entity_player_start
        .and_then(|e| e.get_property("origin"))
        .map(|origin| {
            let mut parts = origin.split_whitespace().flat_map(str::parse::<f32>);
            vec3(parts.next().unwrap_or(0.0), parts.next().unwrap_or(0.0), parts.next().unwrap_or(0.0))
        })
        .unwrap_or(vec3(0.0, 0.0, 0.0));

    let player_rotation: f32 = entity_player_start
        .and_then(|e| e.get_property("angle"))
        .map(|angle| angle.parse().unwrap())
        .unwrap_or(0.0);

    println!("Player spawn: {:?}", player_spawn);

    let mut settings = GameSettings {
        camera_speed: 200.0,
        camera_rotation_speed: 3.0,
        draw_depth: false,
        show_stats: true,
    };

    // create our camera
    let mut camera = Camera {
        position: player_spawn,
        pitch: 0.0,
        yaw: player_rotation.to_radians(),
        aspect: SCREEN_WIDTH as f32 / SCREEN_HEIGHT as f32,
        fov: 70.0,
        znear: 1.0,
        zfar: 10000.0,
    };

    let mut instant = Instant::now();

    // run the main-loop
    while window.is_open() && !window.is_key_down(Key::Escape) {
        // fps counter
        let elapsed_seconds = instant.elapsed().as_secs_f32();
        instant = Instant::now();

        // clear our buffer to start rendering
        renderer.clear();

        // process input
        process_input(&window, elapsed_seconds, &mut settings, &mut camera);

        let view = camera.get_view_mat();
        let proj = camera.get_projection_mat();

        {
            let world = Mat4::from_scale_rotation_translation(
                Vec3::new(10.0, 10.0, 10.0),
                Quat::IDENTITY,
                Vec3::new(0.0, 0.0, 0.0)
            );

            let wvp = proj * view * world;
            // renderer.draw(&model.vertices, &model.indices, &world, &wvp);
        }

        {
            let world = Mat4::from_scale_rotation_translation(
                Vec3::new(1.0, 1.0, 1.0),
                Quat::IDENTITY,
                Vec3::new(0.0, 0.0, 0.0)
            );

            let wvp = proj * view * world;

            let player_position = camera.position;
            level.draw(&world, &wvp, player_position, &mut renderer);
        }

        let mut buffer: Vec<u32>;

        if settings.draw_depth {
            let depth: &[f32] = renderer.get_depth_buffer();
            buffer = depth.iter()
                .map(|&x| (float_to_u32_color(x)))
                .collect();
        }
        else {
            buffer = Vec::from(renderer.get_back_buffer());
        }

        // draw some informative text
        let text_frame_time = format!("frame: {}ms", (elapsed_seconds * 1000.0) as u32);
        text.draw_text(&mut buffer, 10, 20, text_frame_time.as_str());
        let text_camera_position = format!("camera: {}", camera.position);
        text.draw_text(&mut buffer, 10, 40, text_camera_position.as_str());

        if settings.show_stats {
            let stats = renderer.get_stats();
            let str_stats = format!("{:#?}", stats) // Pretty-prints with new lines
                .trim_start_matches("RendererStats {\n") // Remove struct name
                .trim_end_matches("\n}") // Remove closing brace
                .replace("    ", ""); // Remove excess indentation
            text.draw_text(&mut buffer, 10, 80, str_stats.as_str());

            // let mut y_offset = 100;
            // for (key, value) in &stats.debug_values {
            //     let str = format!("{}: {}", key, value);
            //     text.draw_text(&mut buffer, 10, y_offset, str.as_str());
            //     y_offset += 20;
            // }
        }

        window
            .update_with_buffer(buffer.as_slice(), SCREEN_WIDTH, SCREEN_HEIGHT)
            .unwrap();
    }
}
