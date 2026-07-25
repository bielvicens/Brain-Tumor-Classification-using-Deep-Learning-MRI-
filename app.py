import yaml
import torch
import torch.nn.functional as F

from PIL import Image
import gradio as gr
from torchvision import transforms

from src.model import BrainTumorModel


############################################################
# CONFIGURATION
############################################################

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

img_size = config["training"]["img_size"]
classes = config["classes"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_DESCRIPTIONS = {
    "Glioma": "Tumor originating in glial cells",
    "Meningioma": "Tumor of the meninges",
    "No Tumor": "No tumor detected",
    "Pituitary": "Tumor of the pituitary gland",
}

############################################################
# MODEL
############################################################

model = BrainTumorModel(num_classes=4)

model.load_state_dict(
    torch.load(
        "models/best_model.pth",
        map_location=device,
    )
)

model.to(device)
model.eval()

############################################################
# TRANSFORMS
############################################################

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

############################################################
# PREDICTION & HTML GENERATION
############################################################

# Estat inicial buit perquè l'UI quedi elegant abans de predir
INITIAL_FINDING_HTML = """
<div style="border:2px dashed #cbd5e1; background:#f8fafc; border-radius:12px; 
            padding:40px 20px; text-align:center; color:#64748b; margin-bottom:16px;">
    <svg style="width:40px; height:40px; margin:0 auto 10px auto; opacity:0.5;" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
    </svg>
    <div style="font-family:'Inter', sans-serif; font-size:0.95rem; font-weight:500;">
        Waiting for MRI scan...
    </div>
    <div style="font-family:'Inter', sans-serif; font-size:0.8rem; margin-top:4px;">
        Upload an image and click analyze to generate the report.
    </div>
</div>
"""

def build_finding_html(top_class, confidence):
    # Canviem el color depenent de si és tumor o no
    accent_color = "#10b981" if top_class == "No Tumor" else "#ef4444"
    bg_color = "#ecfdf5" if top_class == "No Tumor" else "#fef2f2"
    
    return f"""
    <div style="border:1px solid {accent_color}40; border-left:6px solid {accent_color};
                background:{bg_color}; border-radius:10px; padding:20px;
                margin-bottom:20px; box-shadow: 0 2px 10px rgba(0,0,0,0.03);">
        <div style="font-size:0.75rem; font-weight:700; letter-spacing:1px;
                    text-transform:uppercase; color:#475569; margin-bottom:8px;">
            Primary Finding
        </div>
        <div style="font-size:1.6rem; font-weight:800; color:#0f172a;
                    font-family:'Inter', sans-serif; margin-bottom:6px;">
            {top_class}
        </div>
        <div style="font-size:0.9rem; font-weight:500; color:#475569;
                    font-family:'Inter', sans-serif; display:flex; align-items:center;">
            <span style="background:#ffffff; padding:4px 10px; border-radius:20px; 
                         border:1px solid #cbd5e1; margin-right:10px; font-weight:600; color:#0f172a">
                {confidence:.1f}% Confidence
            </span>
            <span style="color: #475569;">AI Diagnostic Model</span>
        </div>
    </div>
    """

def predict(image):
    if image is None:
        return None, INITIAL_FINDING_HTML

    image = image.convert("RGB")
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

    probs = {}
    for i, p in enumerate(probabilities):
        probs[classes[i]] = float(p)

    top_class = max(probs, key=probs.get)
    confidence = probs[top_class] * 100

    finding_html = build_finding_html(top_class, confidence)

    return probs, finding_html


############################################################
# CLINICAL STYLE (CSS & THEME)
############################################################

CLINICAL_CSS = """
/* Reset & Fonts */
.gradio-container {
    font-family: 'Inter', system-ui, sans-serif !important;
    max-width: 1100px !important;
    margin: auto !important;
}

/* Main Header Card */
#app-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border-radius: 16px;
    padding: 35px 40px;
    margin-bottom: 30px;
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.15);
    border: 1px solid #334155;
}

.header-eyebrow {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #38bdf8 !important; /* Light blue accent */
    margin-bottom: 8px !important;
}

#app-header h1 {
    color: #ffffff !important;
    font-weight: 700;
    font-size: 2rem !important;
    margin-bottom: 10px !important;
    letter-spacing: -0.5px;
}

.header-sub {
    color: #94a3b8 !important;
    font-size: 1.05rem !important;
    line-height: 1.5;
}

/* Column Cards */
.clinic-card {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 16px !important;
    padding: 24px !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important;
}

.card-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #0f172a;
    border-bottom: 2px solid #f1f5f9;
    padding-bottom: 12px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Remove default gradio background inside empty image upload */
.gradio-image {
    background-color: #f8fafc !important;
    border-radius: 12px !important;
}

/* Main Button */
#predict-btn {
    background: #2563eb !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px;
    font-size: 1rem !important;
    border-radius: 10px !important;
    padding: 12px 0 !important;
    margin-top: 15px !important;
    transition: all 0.2s ease-in-out !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25) !important;
}

#predict-btn:hover {
    background: #1d4ed8 !important;
    box-shadow: 0 6px 16px rgba(37, 99, 235, 0.35) !important;
    transform: translateY(-1px);
}

/* Accordion */
.gradio-container .accordion {
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    background: #f8fafc !important;
}

footer {
    display: none !important;
}
"""

clinical_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#f1f5f9",
    block_background_fill="#ffffff",
    block_border_color="#e2e8f0",
    block_radius="12px",
    body_text_color="#334155",
)

############################################################
# STATIC HTML BLOCKS
############################################################

REFERENCE_TABLE_HTML = """
<table style="width:100%; border-collapse:collapse; font-family:'Inter', sans-serif;">
    <thead>
        <tr>
            <th style="text-align:left; background:#f1f5f9; color:#475569;
                       padding:12px; font-size:0.75rem; text-transform:uppercase;
                       letter-spacing:1px; border-radius: 8px 0 0 0;">Category</th>
            <th style="text-align:left; background:#f1f5f9; color:#475569;
                       padding:12px; font-size:0.75rem; text-transform:uppercase;
                       letter-spacing:1px; border-radius: 0 8px 0 0;">Clinical Description</th>
        </tr>
    </thead>
    <tbody>
"""

for i, (name, desc) in enumerate(CLASS_DESCRIPTIONS.items()):
    row_bg = "#ffffff" if i % 2 == 0 else "#f8fafc"
    REFERENCE_TABLE_HTML += f"""
        <tr style="background:{row_bg};">
            <td style="padding:12px; color:#0f172a; font-weight:600;
                       border-bottom:1px solid #e2e8f0; font-size:0.85rem;">{name}</td>
            <td style="padding:12px; color:#64748b;
                       border-bottom:1px solid #e2e8f0; font-size:0.85rem;">{desc}</td>
        </tr>
    """

REFERENCE_TABLE_HTML += "</tbody></table>"

DISCLAIMER_HTML = """
<div style="border:1px solid #e2e8f0; border-left:4px solid #94a3b8;
            background:#f8fafc; border-radius:8px; padding:16px;
            margin-top:20px;">
    <div style="font-size:0.75rem; font-weight:700; letter-spacing:0.5px;
                text-transform:uppercase; color:#475569; margin-bottom:6px;">
        ⚠️ Research Use Only
    </div>
    <div style="font-size:0.85rem; color:#64748b; line-height:1.5;">
        This Deep Learning model is an experimental support tool. It does not replace 
        professional radiological assessment or clinical diagnosis.
    </div>
</div>
"""

############################################################
# INTERFACE
############################################################

with gr.Blocks(theme=clinical_theme, css=CLINICAL_CSS, title="Brain Tumor Classification") as demo:

    # --- Header Superior ---
    with gr.Column(elem_id="app-header"):
        gr.Markdown("Medical AI Support System", elem_classes="header-eyebrow")
        gr.Markdown("# Brain MRI Classification")
        gr.Markdown(
            "Automated analysis of magnetic resonance imaging for neuro-oncological screening. "
            "Powered by ResNet50 Transfer Learning.",
            elem_classes="header-sub",
        )

    # --- Cos Principal ---
    with gr.Row():

        # Columna Esquerra: INPUT
        with gr.Column(scale=1, elem_classes="clinic-card"):
            gr.HTML("<div class='card-title'> 1. Patient Imaging</div>")
            
            # show_label=False és la clau perquè no es barregin els textos lletjos de Gradio
            image_input = gr.Image(
                type="pil",
                show_label=False, 
                height=350,
                elem_id="image-input"
            )
            predict_btn = gr.Button("Analyze Image", elem_id="predict-btn")

        # Columna Dreta: OUTPUT
        with gr.Column(scale=1, elem_classes="clinic-card"):
            gr.HTML("<div class='card-title'>2. Analysis Report</div>")

            # El requadre on sortirà el text de resultat. 
            # Hi posem l'estat "Waiting for MRI..." per defecte.
            finding_html = gr.HTML(value=INITIAL_FINDING_HTML)

            # Barres de probabilitat, sense l'etiqueta default de Gradio
            output_label = gr.Label(
                num_top_classes=4,
                show_label=False,
            )

            # Acordió d'informació
            with gr.Accordion("View Category Reference", open=False):
                gr.HTML(REFERENCE_TABLE_HTML)

            # Avís Legal
            gr.HTML(DISCLAIMER_HTML)

    # Lògica del botó
    predict_btn.click(
        fn=predict, 
        inputs=image_input, 
        outputs=[output_label, finding_html]
    )

############################################################

if __name__ == "__main__":
    demo.launch()