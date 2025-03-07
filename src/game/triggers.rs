use crate::entity::{Entity, EntityData, SolidType};

#[derive(Default)]
pub struct TriggerOnce {

}

impl Entity for TriggerOnce {
    fn construct(&mut self, data: &mut EntityData) {
        data.solid = SolidType::Trigger;
    }
}

impl TriggerOnce {

}

#[derive(Default)]
pub struct TriggerMultiple {

}

impl Entity for TriggerMultiple {
    fn construct(&mut self, data: &mut EntityData) {
        data.solid = SolidType::Trigger;
    }
}

impl TriggerMultiple {
    
}