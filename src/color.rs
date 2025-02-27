use glam::Vec4;

#[derive(Debug, Copy, Clone)]
pub struct Color {
    rgba: Vec4,
}

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
