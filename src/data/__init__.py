from pydantic import BaseModel
# from ..utils import BaseUnionModel
from fastapi import HTTPException
from typing import Optional, List, Dict
import pandas as pd

class Item(BaseModel):
    school: Optional[str] = None
    sex: Optional[str] = None
    age: int
    address: int
    famsize: int
    Pstatus: int
    Medu: int
    Fedu: int
    Mjob: Optional[str] = None
    Fjob: Optional[str] = None
    reason: Optional[str] = None
    guardian: Optional[str] = None
    studytime: int
    failures: Optional[int] = None
    schoolsup: int
    famsup: int
    paid: int
    activities: int
    nursery: Optional[int] = None
    higher: Optional[int] = None
    internet: int
    romantic: int
    famrel: int
    freetime: int
    goout: int
    Dalc: int
    Walc: int
    health: Optional[int] = None
    abscences: Optional[int] = None
    traveltime: int

    G3: Optional[int] = None

def process_data(data:Dict[str, Item], cols: Optional[List[str]] = None):
    ids = list(data.keys())
    data = list(data.values())
    df = pd.DataFrame(list(map(lambda x: x.dict(), data))).dropna(axis=1)
    if cols is not None:
        try:
            if 'G3' in df:
                cols.append('G3')
            df = df[cols]
        except:
            raise HTTPException(
                status_code=404,
                detail="Features cannot be found in data. Check `cols` and `data`."
            )

    if 'G3' in df:
        x = df.drop('G3', axis=1)
        y = df['G3']
        return x, y, ids
    return df, None, ids