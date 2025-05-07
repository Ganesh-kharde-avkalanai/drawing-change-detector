## How to Run the Project Locally
### Prerequisites
- Python3 installed
- API key for Google Gemini

### Steps to Set Up and Run
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Ganesh-kharde-avkalanai/drawing-change-detector.git
   cd drawing-change-detector
   ```


2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Update the Gemini API key**:
   ```bash
   # Set your Gemini API key
    genai.configure(api_key="Paste API Key") 
   ```
   

4. **Run the Flask server**:
   ```bash
   streamlit run app.py
   ```

   or
   ```bash
   python -m streamlit run app.py
   ```
