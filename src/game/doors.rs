use crate::entity::{Entity, EntityData, SolidType};

#[derive(Default)]
pub struct FuncDoor {

}

impl Entity for FuncDoor {
    fn construct(&mut self, data: &mut EntityData) {
        data.solid = SolidType::BSP;
    }
}

impl FuncDoor {
    
}