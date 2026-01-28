# Custom-ICON-CH1-Data-Fetch

This project fetches and visualizes weather data from the ICON-CH1 model.

## 🚀 Features
- Fetches real-time data from ICON-CH1 sources.
- Generates Skew-T plots for specific locations.
- Automated daily execution via GitHub Actions.

## 🛠️ Installation
```bash
pip install -r requirements.txt
📈 Usage
Configure Locations: Edit locations.json to add your coordinates.

Fetch Data: Run python fetch_data.py

Plotting: Run python plot_skewt.py

🤖 Automation
The project uses GitHub Actions (see .github/workflows/daily_plot.yml) to automatically update plots once a day.
