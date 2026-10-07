# AI-Based Road Accident Risk Prediction & Safe Route Recommendation Platform

A Final Year Capstone Project built for a B.Tech degree in Computer Science and Engineering. This platform uses Machine Learning and Graph Theory to predict road accident risks in real-time and recommend the safest paths for drivers, instead of just the fastest ones.

---

## 🎓 Academic Affiliation
* **Institution:** BVC Engineering College, Odalarevu
* **Department:** Computer Science and Engineering (CSE)
* **Project Type:** B.Tech Final Year Major Project

---

## 🌍 The Real-World Problem It Solves
Standard navigation apps like Google Maps or Apple Maps always show you the **fastest** or **shortest** route to your destination. However, they are completely blind to road safety. 

Certain road segments become highly dangerous under specific conditions—such as driving through a dark curve at 11:00 PM during heavy rain, or crossing a high-congestion intersection during peak office hours. Taking the fastest route during these times exponentially increases the chances of a car accident. 

---

## 💡 The AI Solution
This project builds a smart navigation platform that prioritizes **passenger safety over minor speed trade-offs**. 

1. **The AI Brain (XGBoost/RandomForest):** The system takes historical accident data and trains an AI model. When a user inputs real-time environmental factors (like Weather: Rain, Time: 9:00 PM, Traffic: High), the AI instantly calculates a dynamic "Accident Risk Score" (from 0% to 100%) for different road paths.
2. **The Smart Routing Engine (NetworkX & Dijkstra):** The project models a city road map as a connected digital network grid of nodes (intersections) and edges (roads). We modified the classic Dijkstra routing algorithm. If the AI detects a high accident risk on a road segment, the algorithm penalizes that road and dynamically reroutes the driver through a safer, alternative path.

---

## 🛠️ How the Code Architecture Works
The project is split into clean, modular files that are easy to present during college audits:
* `real_accident_data.csv`: A large dataset containing historical records matching weather, traffic, and lighting conditions to accident risk.
* `train_model.py`: Reads the raw dataset, performs feature engineering, trains the Machine Learning model, and saves it as a binary file (`accident_model.pkl`).
* `routing_engine.py`: Builds a 15-node urban intersection map grid and runs the customized, risk-penalized shortest path algorithm.
* `app.py`: The user interface dashboard. It creates the input sidebars, calculates comparison tables (Fastest vs. Safest), and renders the paths on an interactive web map.

---

## 💻 Technologies Used
* **Core Language:** Python 3.10+
* **Machine Learning Pipelines:** XGBoost, Scikit-Learn (For predicting risk scores)
* **Graph Mathematics:** NetworkX (For building the city road grid and calculating paths)
* **Geospatial UI Map Rendering:** Folium, Streamlit-Folium, OpenStreetMap (For the interactive map screen)
* **Data Scaffolding:** Pandas, NumPy
* **Frontend Dashboard:** Streamlit (For the user interface web layout)
* **Automation Scripts:** Native Command Batch Files (`.bat`, `.sh`) for instant startup

---

## 🚀 How to Run the Project (Deployment Lifecycle)
You can launch this complete project locally on your system using three simple steps in your terminal window:

### 1. Install Libraries
Create the virtual runtime environment and download all the required Python packages:
```powershell
.\run_setup.bat
```

### 2. Train the AI Model
Process the data tables, train the machine learning algorithm, and compile the model brain:
```powershell
.\run_train.bat
```

### 3. Launch the Interactive Dashboard
Deploy the local web server and open the live map tracking dashboard automatically in your browser:
```powershell
.\run_app.bat
```

---

## 📊 Project Output Analysis
When you inject hazardous parameters into the sidebar (e.g., matching a heavy rainstorm with nighttime driving conditions), the app dynamically shows its value:
* **The Shortest Path (Solid Red Line on Map):** Shows the default fastest route, highlighting a high calculated accident risk score (e.g., 75%).
* **The Safest Path (Solid Green Line on Map):** Intelligently guides the driver around the dangerous bottleneck areas, achieving a **+42% safety gain** while keeping the total travel distance extension under a minimal 8% variation threshold.
