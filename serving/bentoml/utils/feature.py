from pydantic import BaseModel
from typing import List


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
    chest: float
    waist: float
    hip: float
    thigh: float
    arm: float
    leg: float
    shoulder: float


class Product_Input(BaseModel):
    product_ids: List[int] = [1, 2, 3]
    gender: str = "M"


class Product_Output(BaseModel):
    product_ids: List[int] = [146, 200, 267, 6, 345]


class UserId(BaseModel):
    user_id: int = 1


class NewProductId(BaseModel):
    product_ids: List[int] = [1, 2, 3]
