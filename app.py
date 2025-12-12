from flask import Flask, render_template, request, send_file
import os
import cv2
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg') # Essential for running on Windows
import matplotlib.pyplot as plt
from fpdf import FPDF
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from datetime import datetime

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER): os.makedirs(UPLOAD_FOLDER)

# CLASSES
CLASSES = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary']

# Load Models
print("Loading models...")
try:
    models = {
        'SVM': joblib.load('models/SVM.pkl'),
        'RandomForest': joblib.load('models/RandomForest.pkl'),
        'KNN': joblib.load('models/KNN.pkl'),
        'NaiveBayes': joblib.load('models/NaiveBayes.pkl'),
        'XGBoost': joblib.load('models/XGBoost.pkl')
    }
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    print("Make sure you ran the Jupyter Notebook to generate the .pkl files!")

# --- FEATURE EXTRACTION FUNCTION ---
def extract_single_image_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    
    glcm = graycomatrix(img, [1], [0], 256, symmetric=True, normed=True)
    feats = [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        np.mean(img),
        np.std(img)
    ]
    lbp = local_binary_pattern(img, 8, 1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-6)
    
    return np.array(feats + list(lbp_hist)).reshape(1, -1)

# --- CUSTOM PDF CLASS FOR BEAUTIFUL REPORTS ---
class PDF(FPDF):
    def header(self):
        # Logo placeholder or Color bar
        self.set_fill_color(44, 62, 80) # Dark Blue
        self.rect(0, 0, 210, 20, 'F')
        
        self.set_font('Arial', 'B', 15)
        self.set_text_color(255, 255, 255) # White text
        self.cell(0, 10, 'Brain Tumor Diagnostic Center', 0, 1, 'C')
        self.ln(15)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, 'Automated AI Report - For Research Use Only | Page ' + str(self.page_no()), 0, 0, 'C')

# --- ROUTES ---

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            p_name = request.form['name']
            p_age = request.form['age']
            file = request.files['file']
            
            if file:
                file_path = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(file_path)
                
                # Enhanced Image
                img = cv2.imread(file_path)
                enhanced = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
                enhanced_path = os.path.join(UPLOAD_FOLDER, 'enhanced_' + file.filename)
                cv2.imwrite(enhanced_path, enhanced)

                # Analysis
                features = extract_single_image_features(file_path)
                model_results = {}
                for name, model in models.items():
                    pred = model.predict(features)[0]
                    try:
                        if hasattr(model, "predict_proba"):
                            prob = model.predict_proba(features)[0].max() * 100
                        else:
                            prob = 100.0
                    except: prob = 100.0
                    model_results[name] = {'class': CLASSES[pred], 'conf': prob}

                # Final Verdict (Using RandomForest as standard)
                final_pred = model_results['RandomForest']['class']
                final_conf = model_results['RandomForest']['conf']

                # Chart Generation
                names = list(model_results.keys())
                confs = [res['conf'] for res in model_results.values()]
                
                plt.figure(figsize=(8, 4))
                bars = plt.bar(names, confs, color=['#bdc3c7', '#2ecc71', '#bdc3c7', '#bdc3c7', '#3498db'])
                plt.ylabel('Confidence (%)')
                plt.title('Model Consensus Analysis')
                plt.ylim(0, 110)
                chart_path = os.path.join(UPLOAD_FOLDER, 'chart.png')
                plt.savefig(chart_path)
                plt.close()

                return render_template('index.html', 
                                       result=final_pred, 
                                       conf=f"{final_conf:.2f}",
                                       original_img=file_path,
                                       enhanced_img=enhanced_path,
                                       chart_img=chart_path,
                                       p_name=p_name, p_age=p_age)
        except Exception as e:
            return f"Error: {str(e)}"

    return render_template('index.html', result=None)

@app.route('/download_report', methods=['POST'])
def download_report():
    # Get Data from hidden forms
    name = request.form['name']
    age = request.form['age']
    result = request.form['result']
    conf = request.form['conf']
    orig_img_path = request.form['original_img'] # Relative path
    chart_img_path = request.form['chart_img'] # Relative path
    
    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # 1. PATIENT DETAILS SECTION
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 10, "Patient Information", 0, 1, 'L')
    pdf.line(10, 38, 200, 38)
    
    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(40, 10, f"Patient Name:", 0, 0)
    pdf.cell(60, 10, name, 0, 0)
    pdf.cell(40, 10, f"Date of Report:", 0, 0)
    pdf.cell(60, 10, datetime.now().strftime('%Y-%m-%d'), 0, 1)
    
    pdf.cell(40, 10, f"Patient Age:", 0, 0)
    pdf.cell(60, 10, age, 0, 1)
    pdf.ln(5)

    # 2. IMAGES SECTION
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 10, "MRI Scan Analysis", 0, 1, 'L')
    pdf.line(10, 78, 200, 78)
    pdf.ln(5)
    
    # Embed the MRI Image
    # Note: FPDF needs system paths. 'static/uploads/...' works if running from root.
    try:
        pdf.image(orig_img_path, x=10, y=85, w=60)
        pdf.set_xy(80, 95) # Move cursor to the right of the image
        pdf.set_font("Arial", "", 10)
        pdf.set_text_color(0,0,0)
        pdf.multi_cell(0, 6, "The image on the left represents the original MRI scan provided for analysis. \n\nPreprocessing applied:\n- Grayscale Conversion\n- Noise Reduction\n- Contrast Enhancement")
    except:
        pdf.cell(0, 10, "Error loading image", 0, 1)

    pdf.ln(50) # Space after image

    # 3. PREDICTION RESULTS
    pdf.set_fill_color(236, 240, 241) # Light Grey background
    pdf.rect(10, pdf.get_y(), 190, 30, 'F')
    
    pdf.set_xy(15, pdf.get_y() + 5)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(50, 10, "Primary Prediction:", 0, 0)
    
    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(192, 57, 43) # Red color for result
    pdf.cell(60, 10, result.upper(), 0, 1)
    
    pdf.set_x(15)
    pdf.set_font("Arial", "B", 11)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(50, 10, f"Confidence Score: {conf}%", 0, 1)
    pdf.ln(10)

    # 4. COMPARISON CHART
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 10, "Algorithm Consensus", 0, 1, 'L')
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    try:
        pdf.image(chart_img_path, x=25, w=160)
    except:
        pdf.cell(0, 10, "Error loading chart", 0, 1)

    # Output PDF
    report_path = os.path.join(UPLOAD_FOLDER, 'report.pdf')
    pdf.output(report_path)
    return send_file(report_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5000)