<h1 align="center">
  <br>
  Car Image Analysis Project
</h1>

<div align="center">
  <h4>
    <a href="#overview">Overview</a> |
    <a href="#getting-started">Getting Started</a> |
    <a href="#models-and-apis">Models and APIs</a> |
    <a href="#additional-information">Additional Information</a>
  </h4>
</div>

<br>

# Overview
This project provides a realistic assessment of vehicle damage, addressing the need for insurance companies to accurately verify damage before approving coverage for repairs. The system first distinguishes between real and AI-generated images, ensuring authenticity. It then evaluates the severity of the damage, classifying it as Minor, Moderate, or Severe, followed by detecting the specific damaged area of the car.

In some stages, multiple models were tested and compared to select the most effective one for the task.



# Getting Started

Follow these steps to set up and run the Car Image Analysis Streamlit App:

1. Clone the repository:

   ```bash
   git clone https://github.com/Aymen568/Car-Image-Analysis-Project.git
2. Install the required dependencies:
    ```bash
   pip install -r requirements.txt
3. Run the Streamlit app :
    ```bash
    streamlit run streamlit_app.py
4. Open your browser and navigate to the provided local URL to access the app.



# Models and APIs

The application employs various models and APIs:
- **AI Image Detection:**
  - A custom model determines whether an image is AI-generated.
    - Trained models:
      - Convolutional Neural Network (CNN) model from scratch
      - Inception V3 trained on [AIData Kaggle dataset](https://www.kaggle.com/datasets/derrickmwiti/aidata)
      - Vision Transformer (VIT) trained on the CIFake dataset
    - Best performing model: Inception V3

- **Damage Severity Assessment:**
  - A custom model assesses the severity of car damage using the [Car Damage Severity Dataset](https://www.kaggle.com/datasets/prajwalbhamere/car-damage-severity-dataset).
    - Trained models:
      - ResNet
      - 4D model
      - EfficientNet 
    - Best performing model: EfficientNet

- **Damaged Parts Detection:**
  - YOLOv8, trained on the Car Damage COCO Dataset [Car Damage COCO Dataset](https://universe.roboflow.com/dan-vmm5z/car-damage-coco-dataset),  identifies the damaged areas of the vehicle.


# Additional Information
- For more details on each algorithm, model training, and external dependencies, refer to the specific sections in the source code.
