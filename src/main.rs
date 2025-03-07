mod model;
mod level;
mod camera;
mod renderer;
mod engine;
mod backbuffer;
mod math;
mod entity;
mod bsp;
mod game;

use crate::engine::{DebugStats, Engine};
use crate::level::Level;
use crate::renderer::{Renderer, RendererSettings};

use crate::backbuffer::BackBuffer;
use crate::camera::CameraSettings;
use camera::Camera;
use glam::{vec2, vec3, Mat4, Quat, Vec3};
use minifb::{Key, Scale, Window, WindowOptions};
use minifb_fonts::font6x8;
use crate::math::Color;

const SCREEN_WIDTH: u32 = 1920/2;
const SCREEN_HEIGHT: u32 = 1080/2;

struct GameSettings {
    pub draw_depth: bool,
    pub show_stats: bool,
}

fn float_to_u32_color(value: f32) -> u32 {
    let intensity = (value.clamp(0.0, 1.0) * 255.0).round() as u32;
    0xFF000000 | (intensity << 16) | (intensity << 8) | intensity
}

fn process_input(window: &Window, game_settings: &mut GameSettings, camera: &mut Camera) {
    {
        let mut movement = vec3(0.0, 0.0, 0.0);
        if window.is_key_down(Key::W) {
            movement.x += 1.0;
        }
        if window.is_key_down(Key::S) {
            movement.x -= 1.0;
        }
        if window.is_key_down(Key::A) {
            movement.y += 1.0;
        }
        if window.is_key_down(Key::D) {
            movement.y -= 1.0;
        }
        if window.is_key_down(Key::E) {
            movement.z += 1.0;
        }
        if window.is_key_down(Key::Q) {
            movement.z -= 1.0;
        }

        let mut rotation = vec2(0.0, 0.0);

        if window.is_key_down(Key::Left) {
            rotation.x += 1.0;
        }
        if window.is_key_down(Key::Right) {
            rotation.x -= 1.0;
        }
        if window.is_key_down(Key::Up) {
            rotation.y -= 1.0;
        }
        if window.is_key_down(Key::Down) {
            rotation.y += 1.0;
        }

        camera.set_input(movement, rotation);
    }

    if window.is_key_released(Key::Tab) {
        game_settings.draw_depth = !game_settings.draw_depth;
    }
}

fn main() {
    // create our renderer
    let renderer = Renderer::new(RendererSettings {
        naive_rasterization: true,
        tile_size: 16,
        wireframe: false,
        ..RendererSettings::default()
    });

    let mut engine: Engine = Engine::new(renderer);

    let mut stats: DebugStats = DebugStats::default();

    // create a text renderer
    let text_color: Color = Color::from_u8(255,255,255,255);
    let text = font6x8::new_renderer(SCREEN_WIDTH as usize, SCREEN_HEIGHT as usize, text_color.to_u32());

    // create a window and buffer
    let mut window = Window::new(
        "RustedQuake",
        SCREEN_WIDTH as usize,
        SCREEN_HEIGHT as usize,
        WindowOptions {
            scale: Scale::X2,
            ..WindowOptions::default()
        }
    ).unwrap_or_else(|e| {
        panic!("{}", e);
    });

    // load our test obj file
    // let model = Model::load("data/plane.obj").expect("Failed to load model");
    let mut level = Level::load("maps/e1m1_remaster.bsp").expect("Failed to load level");

    let info_player_start = level.get_entity_of_class_name("info_player_start");
    let (player_spawn, player_rotation) = info_player_start
        .map(|entity| (entity.origin, entity.angle))
        .unwrap_or((Vec3::ZERO, 0.0));

    let mut settings = GameSettings {
        draw_depth: false,
        show_stats: true,
    };

    // create our camera
    let mut camera = Camera::new(
        CameraSettings {
            aspect: SCREEN_WIDTH as f32 / SCREEN_HEIGHT as f32,
            fov: 70.0,
            znear: 0.1,
            zfar: 2500.0,
        },
    );

    camera.set_position(player_spawn);
    camera.set_rotation(player_rotation, 0.0);

    let mut buffer = BackBuffer::new(SCREEN_WIDTH, SCREEN_HEIGHT);

    // run the main-loop
    while window.is_open() && !window.is_key_down(Key::Escape) {
        engine.update();
        stats.clear();

        // clear our buffer to start rendering
        engine.renderer().clear(&mut buffer);

        // process input
        process_input(&window, &mut settings, &mut camera);

        let view = camera.get_view_mat();
        let proj = camera.get_projection_mat();

        camera.update(&engine, &level);

        {
            let world = Mat4::from_scale_rotation_translation(
                Vec3::new(1.0, 1.0, 1.0),
                Quat::IDENTITY,
                Vec3::new(0.0, 0.0, 0.0)
            );

            let wvp = proj * view * world;

            let player_position = camera.position;
            level.update_visibility(player_position);
            level.draw(&engine, &mut stats, &mut buffer, player_position, &world, &view, &proj, &wvp);
        }

        let mut buf: Vec<u32>;

        if settings.draw_depth {
            let depth: &[f32] = buffer.get_depth_buffer();
            buf = depth.iter()
                .map(|&x| (float_to_u32_color(x)))
                .collect();
        }
        else {
            buf = Vec::from(buffer.get_back_buffer());
        }

        // draw some informative text
        let text_frame_time = format!("frame: {}ms", (engine.time().elapsed_time() * 1000.0) as u32);
        text.draw_text(&mut buf, 10, 20, text_frame_time.as_str());
        let text_camera_position = format!("camera: {}", camera.position);
        text.draw_text(&mut buf, 10, 40, text_camera_position.as_str());

        if settings.show_stats {
            let str_stats = format!("{:#?}", stats) // Pretty-prints with new lines
                .trim_start_matches("RendererStats {\n") // Remove struct name
                .trim_end_matches("\n}") // Remove closing brace
                .replace("    ", ""); // Remove excess indentation
            text.draw_text(&mut buf, 10, 80, str_stats.as_str());

            // let mut y_offset = 100;
            // for (key, value) in &stats.debug_values {
            //     let str = format!("{}: {}", key, value);
            //     text.draw_text(&mut buffer, 10, y_offset, str.as_str());
            //     y_offset += 20;
            // }
        }

        window
            .update_with_buffer(buf.as_slice(), buffer.get_width() as usize, buffer.get_height() as usize)
            .unwrap();
    }
}
