## Predictive Maintenance for Advanced Reverse Osmosis Systems with Artificial Intelligence

**Maturity Level**: TRL 2-3

### Description
This project aims to develop an artificial intelligence system for predictive maintenance of advanced reverse osmosis systems. Using supervised learning techniques and machine learning models, it seeks to predict membrane wear and optimize the process.

Our planet contains 1,386 million km³ of water, a quantity that has remained unchanged for the last 2 billion years. Of this water, 97% is salty, and only 2.5% is freshwater. In Mexico, the population has grown significantly, reaching 129 million in 2024, exacerbating water scarcity. Among the emerging solutions is reverse osmosis, a process that purifies water by removing salts and contaminants through a semipermeable membrane, making it ideal for desalinating seawater. This project aims to optimize its high cost and maintenance through an artificial intelligence model that predicts the wear of key components by analyzing historical data such as pressure, flow rate, temperature, and salinity, enabling preventive intervention.

### Features
- Prediction of the physical condition of semipermeable membranes.
- Optimization of the desalination process.
- Use of operational historical data: pressure, flow, temperature, and salinity.
- Classification and regression models based on Random Forest.

### Requirements
- Python 3.8+
- Libraries: `scikit-learn`, `pandas`, `numpy`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/mcortezv
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
1. Load the dataset in CSV format into the `data/` directory.
2. Run the main script to train and evaluate the models:
   ```bash
   python main.py
   ```

### Contribution
Please review the [CONTRIBUTING](./CONTRIBUTING.md) file for more details.

### License
This project is licensed under the [MIT License](./LICENSE).