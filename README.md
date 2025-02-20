# EtheronAI API and Codebase

Welcome to EtheronAI! This repository contains the codebase and API for our AI-powered functionalities. Below are instructions to get started.

---

## **Installation**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/EtheronAI/EtheronAI.git
   cd EtheronAI
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask Application**:
   ```bash
   python app.py
   ```

4. **Access the API**:
   The API will be available at [http://localhost:5000](http://localhost:5000).

   Swagger documentation: [http://localhost:5000/apidocs/](http://localhost:5000/apidocs/).

---

## **Features**

1. **Signal Filtering**  
   Filters noise from time-series data using a pre-trained deep learning model.  
   **Endpoint**: `/signal_filtering/filter`

2. **Distributed Intelligent Network**  
   Simulates a blockchain-based decentralized network.  
   **Endpoints**: `/blockchain/mine`, `/blockchain/nodes/add`, `/blockchain/chain`

3. **Adaptive Learning**  
   Implements online learning for continuous model improvement.  
   **Endpoints**: `/adaptive_learning/init`, `/adaptive_learning/update`, `/adaptive_learning/predict`

4. **Cross-Platform Narrative Generation (Coming Soon)**  
   Generates multilingual content using the DeepSeek-R1 model (currently under fine-tuning).

---

## **API Documentation**

For detailed API documentation, visit: [https://api.etheron.ai/apidocs/](https://api.etheron.ai/apidocs/)

---

## **Contribution**

We welcome contributions! If you'd like to contribute, please:

- Fork the repository.
- Create a new branch for your feature/bugfix.
- Submit a pull request.

---

## **Contact**

For questions or feedback, reach out to us at:  
📧 Email: [support@etheron.ai](mailto:support@etheron.ai)  
🌐 Website: [https://etheron.ai](https://etheron.ai)

---

## **License**

This project is licensed under the MIT License. See the LICENSE file for details.
