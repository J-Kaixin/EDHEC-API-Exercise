from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# 创建 FastAPI 实例
app = FastAPI()

# ✅ 加载训练好的模型
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# ✅ 定义输入数据模型（InputPayload）
class InputPayload(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# ✅ 定义输出数据模型（OutputPayload）
class OutputPayload(BaseModel):
    species: str  # 返回的花卉种类名称

# ✅ 定义数值预测值到花卉种类的映射
SPECIES_MAPPING = {0: "setosa", 1: "versicolor", 2: "virginica"}

# ✅ 定义 API 端点
@app.post("/predict", response_model=OutputPayload)  # 确保返回的 JSON 结构符合 OutputPayload
async def predict(payload: InputPayload):
    # 1️⃣ 处理输入数据：转换为 NumPy 数组
    data = np.array([[payload.sepal_length, payload.sepal_width, payload.petal_length, payload.petal_width]])

    # 2️⃣ 进行预测
    prediction = model.predict(data)  # 返回 0, 1, 或 2

    # 3️⃣ 映射到对应的花卉种类
    species = SPECIES_MAPPING[int(prediction[0])]

    # 4️⃣ 返回 JSON 响应
    return OutputPayload(species=species)
