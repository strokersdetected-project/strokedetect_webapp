import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

# ----------------------------
# 1️⃣ โหลด ResNet50
# ----------------------------
model = models.resnet50(pretrained=False)

# เปลี่ยน layer สุดท้ายให้ตรงกับจำนวน class
# สมมติคุณมี 2 class (Normal / Stroke)
model.fc = nn.Linear(model.fc.in_features, 3)

# โหลด weight จากไฟล์ของคุณ
model.load_state_dict(
    torch.load("resnet50_stroke_ep10.pth", map_location=torch.device("cpu"))
)

model.eval()

# ----------------------------
# 2️⃣ หน้าเว็บ
# ----------------------------
st.title("Stroke Detection Web Application")
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    key="image_upload"
)



if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ⚠️ ต้องตรงกับตอน train
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1)

    if prediction.item() == 0:
        st.success("Prediction: Normal")
    else:
        st.error("Prediction: Stroke")


