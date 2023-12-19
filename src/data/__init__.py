from pydantic import BaseModel, Field
from typing import Optional, List

class Data(BaseModel):
    features: List[List[float]] = Field(
        description="Data features",
        examples=[[[1.1, 2.4, 3.2], [3.1, 2.3, 4.5]]]
    )
    targets: Optional[List[int]] = Field(
        description="Target values, optional for prediction",
        examples=[[1, 4]]
    )


# def process_data(data:Data, cols: Optional[List[str]] = None):
#     ids = list(data.keys())
#     data = list(data.values())
#     df = pd.DataFrame(list(map(lambda x: x.dict(), data))).dropna(axis=1)
#     if cols is not None:
#         try:
#             if 'G3' in df:
#                 cols.append('G3')
#             df = df[cols]
#         except:
#             raise HTTPException(
#                 status_code=404,
#                 detail="Features cannot be found in data. Check `cols` and `data`."
#             )

#     if 'G3' in df:
#         x = df.drop('G3', axis=1)
#         y = df['G3']
#         return x, y, ids
#     return df, None, ids