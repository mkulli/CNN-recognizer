from flask import Flask, render_template, request, redirect, url_for, flash
import json
from werkzeug.utils import secure_filename
import torch
from torchvision import models, transforms
from PIL import Image
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

try:
    model = models.resnet50(pretrained=True)
    model.eval()
except Exception as e:
    print(f"Ошибка при загрузке модели: {str(e)}")
    model = None

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html', result=None, image_url=None, result_file=None)

@app.route('/upload', methods=['POST'])
def upload_file():
    print("Upload request received")  
    if 'file' not in request.files:
        flash('Файл не был загружен')
        print("No file in request")  
        return redirect(url_for('index'))

    
    file = request.files['file']
    
    if file.filename == '':
        flash('Не выбран файл для загрузки')
        return redirect(url_for('index'))
    
    if not allowed_file(file.filename):
        flash('Неподдерживаемый формат файла. Пожалуйста, загрузите изображение в формате PNG, JPG или BMP.')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        print(f"Processing file: {file.filename}")  
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath)
            print(f"File saved to: {filepath}")  
        except Exception as e:
            print(f"Error saving file: {str(e)}")  
            flash('Ошибка при сохранении файла')
            return redirect(url_for('index'))

        
        try:
            print("Opening image...")  
            image = Image.open(filepath).convert('RGB')
            print("Image opened successfully")  

            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0)
            
            print("Starting classification...")  
            if model is None:
                print("Model is not loaded!")  
                flash('Модель не загружена')
                return redirect(url_for('index'))
                
            with torch.no_grad():
                output = model(input_batch)
                print("Classification completed")  

            
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top_prob, top_cat = torch.topk(probabilities, 1)
            probability = round(top_prob.item() * 100, 2)
            
            with open('imagenet_class_index.json') as f:
                class_idx = json.load(f)
                class_name = class_idx[str(top_cat.item())][1]
            
            result_filename = os.path.splitext(filename)[0] + '_result.txt'
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            with open(result_path, 'w', encoding='utf-8') as f:
                f.write(f"Class: {class_name}\nProbability: {probability}%\n")
            
            return render_template('index.html', 
                               result=f"{class_name} ({probability}%)",
                               image_url=filepath,
                               result_file=result_filename)
        
        except Exception as e:
            flash(f'Ошибка при обработке изображения: {str(e)}')
            return redirect(url_for('index'))
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    if model is None:
        print("Модель не загружена. Проверьте наличие интернет-соединения и повторите попытку.")
    app.run(debug=True)
