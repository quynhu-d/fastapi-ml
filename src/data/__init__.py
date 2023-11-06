from pydantic import BaseModel
from typing import Optional

class Item(BaseModel):
    school: Optional[str] = None
    sex: Optional[str] = None
    age: int
    address: int
    famsize: int
    Pstatus: int
    Medu: int
    Fedu: int
    Mjob: str
    Fjob: str
    reason: str
    guardian: str
    studytime: int
    failures: int
    schoolsup: int
    famsup: int
    paid: int
    activities: int
    nursery: int
    higher: int
    internet: int
    romantic: int
    famrel: int
    freetime: int
    goout: int
    Dalc: int
    Walc: int
    health: int
    abscences: int
    traveltime: int