# Brain Tumor Medical Diagnosis System

A deep learning-based web application for automated brain tumor detection from MRI images.

## Features
- Custom CNN and VGG16 transfer learning models for tumor classification
- Robust image preprocessing (noise cancellation, augmentation)
- Batch and single image prediction via Flask API
- PDF report generation and email delivery
- Modern web frontend: image upload, annotation, progress bar, prediction history

## Project Structure
- `app.py` - Flask backend API
- `brain_tumor_cnn.py` - Model training (CNN & VGG16)
- `brain_tumor_preprocess.py` - Data preprocessing and augmentation
- `upload_test.html` - Web frontend
- `requirements.txt` - Python dependencies
- `dataset/` - MRI images (`yes`/`no` folders)
- `uploads/` - Uploaded images

## Getting Started

1. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

2. **Preprocess data:**
   ```
   python brain_tumor_preprocess.py
   ```

3. **Train model:**
   ```
   python brain_tumor_cnn.py
   ```

4. **Run Flask API:**
   ```
   python app.py
   ```

5. **Open `upload_test.html` in your browser.**

## Notes
- For email/PDF features, configure your email settings in `app.py`.
- Dataset should be placed in the `dataset/yes` and `dataset/no` folders.

## License
This project is for educational and research purposes.
