from pydantic import BaseModel


class ImageS3Path(BaseModel):
    front: str = "0/front.jpg"
    side: str = "0/side.jpg"


class User(BaseModel):
    front: str = "0/front_masked.jpg"
    side: str = "0/side_masked.jpg"
    height: float = 177
    weight: float = 65
    sex: str = "M"


class UserSize(BaseModel):
    height: float
    chest_circumference: float
    waist_circumference: float
    hip_circumference: float
    thigh_left_circumference: float
    arm_left_length: float
    inside_leg_height: float
    shoulder_breadth: float
