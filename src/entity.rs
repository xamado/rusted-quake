use glam::{vec3, Vec3};
use std::collections::HashMap;

#[derive(PartialEq)]
pub enum SolidType {
    Not,
    Trigger,
    BoundingBox,
    SlideBox,
    BSP,
}

pub struct EntityData {
    pub origin: Vec3,
    pub solid: SolidType,
    pub angle: f32,
    pub model_index: i32,
    pub class_name: String,
}

pub trait Entity {
    fn construct(&mut self, _data: &mut EntityData) {}
    
    fn think(&mut self, _data: &mut EntityData) {}
}

#[derive(Default)]
pub struct InfoEntity;

impl Entity for InfoEntity {}

pub fn parse_vec3(properties: &HashMap<String, String>, key: &str) -> Option<Vec3> {
    let str_value = properties.get(key).map_or("", String::as_str);

    let parts: Vec<f32> = str_value
        .split_whitespace()
        .flat_map(str::parse::<f32>)
        .collect();

    if parts.len() == 3 {
        return Some(vec3(parts[0], parts[1], parts[2]));
    }

    None
}

pub fn parse_f32(properties: &HashMap<String, String>, key: &str) -> Option<f32> {
    let str_value = properties.get(key).map_or("", String::as_str);

    str::parse::<f32>(str_value).ok()
}

pub fn parse_model(properties: &HashMap<String, String>, key: &str) -> Option<i32> {
    let str_value = properties.get(key).map_or("", String::as_str);
    let stripped_value = str_value.strip_prefix('*').unwrap_or(str_value);
    stripped_value.parse::<i32>().ok()
}