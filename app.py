from flask import Flask, request, render_template
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import io
import os

app = Flask(__name__)
port = int(os.environ.get("PORT", 8000))  # Use PORT from environment
    app.run(host='0.0.0.0', port=port)
# Define the SwinBlock class (from swint_v2.py)
class SwinBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)

        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        B, C, H, W = x.shape

        x = x.permute(0, 2, 3, 1)
        shortcut = x
        x = self.norm1(x)
        x = x.reshape(B, H * W, C)
        x = x.permute(1, 0, 2)

        attn_out, _ = self.attn(x, x, x)

        attn_out = attn_out.permute(1, 0, 2)
        attn_out = attn_out.reshape(B, H, W, C)
        x = shortcut + attn_out

        shortcut = x
        x = self.norm2(x)
        x = x.reshape(B * H * W, C)
        x = self.mlp(x)
        x = x.reshape(B, H, W, C)
        x = shortcut + x

        x = x.permute(0, 3, 1, 2)

        return x

# Define the SwinTransformerV2 model class (from swint_v2.py)
class SwinTransformerV2(nn.Module):
    def __init__(self, input_resolution=(224, 224), patch_size=4, num_classes=2, embed_dim=96):
        super().__init__()
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.positional_embedding = nn.Parameter(torch.zeros(1, embed_dim, input_resolution[0] // patch_size, input_resolution[1] // patch_size))

        stage1_resolution = (input_resolution[0] // patch_size, input_resolution[1] // patch_size)
        stage2_resolution = (stage1_resolution[0] // 2, stage1_resolution[1] // 2)

        self.stage1 = SwinBlock(embed_dim, stage1_resolution, num_heads=3)
        self.downsample = nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=2, stride=2)
        self.stage2 = SwinBlock(embed_dim * 2, stage2_resolution, num_heads=6)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(embed_dim * 2, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.positional_embedding
        x = self.stage1(x)
        x = self.downsample(x)
        x = self.stage2(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# Load the trained model
def load_model(model_path):
    model = SwinTransformerV2()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Preprocess the image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)
    return image

# Load the model
model_path = 'model.pth'
model = load_model(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detection')
def detection():
    return render_template('detection.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded!', 400

    file = request.files['file']
    
    try:
        # Read and preprocess the image
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        image_tensor = preprocess_image(image)
        
        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
            predicted_class = "Malignant" if predicted.item() == 0 else "Benign"
        
        result = f'Prediction: {predicted_class}'
        return render_template('detection_output.html', result=result)
        
    except Exception as e:
        return f'Error processing image: {str(e)}', 500

if __name__ == '__main__':
    app.run(debug=True)
