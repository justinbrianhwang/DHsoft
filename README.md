# OCR-Powered Automated PDF Survey to Excel System

This project was developed in collaboration with **DHsoft**, a company founded by an honorary professor of Yonsei University. It leverages **optical character recognition (OCR)** and **AI-driven data parsing** to automatically extract survey results from PDF forms and store them in structured Excel spreadsheets.

## Project Overview
![image](https://github.com/user-attachments/assets/e04afb3f-a760-4d7f-b8f2-6c5c8d481816)

- **Project Name**: OCR Survey Automation System
- **Partner Organization**: [DHsoft](https://www.dhsoft.co.kr/)
- **Objective**: To streamline the process of converting handwritten or typed survey responses in PDF format into clean, structured Excel files using OCR and intelligent data mapping.

## Key Features

- 📄 **OCR Extraction**: Automatically extracts handwritten or typed text from PDF-based survey forms.
- 📊 **Excel Conversion**: Maps extracted responses to corresponding columns in an Excel file.
- 🧠 **Intelligent Field Matching**: Uses template-based or AI-assisted matching to accurately associate form entries with predefined Excel headers.
- 🛠️ **Error Detection & Logging**: Identifies malformed entries and logs them for human review.
- 🌐 **Web-based UI (Optional)**: Admin panel for uploading PDFs and downloading Excel results (if applicable in your implementation).

## System Architecture

```
[ PDF Form ]
     ↓ (OCR)
[ Text Extractor (Tesseract / EasyOCR) ]
     ↓
[ Parser & Field Mapper ]
     ↓
[ Excel Generator (pandas + openpyxl) ]
     ↓
[ Exported .xlsx File ]
```

## Installation

### Prerequisites

- Python 3.8+
- pip
- Tesseract OCR (installed locally)
- (Optional) Node.js for frontend

### Installation Steps

```bash
git clone https://github.com/justinbrianhwang/DHsoft.git
cd DHsoft
pip install -r requirements.txt
```

### Running the Script

```bash
# Convert a folder of PDFs to Excel
python main.py --input_dir ./pdfs --output results.xlsx
```

## Tech Stack

- **OCR Engine**: Tesseract OCR / EasyOCR
- **Parsing & Processing**: Python (pandas, re, json)
- **Excel Handling**: openpyxl / xlsxwriter
- **Optional Web UI**: Flask + React

## Example Use Case

1. Researchers distribute printed or fillable PDF surveys.
2. Completed surveys are scanned and uploaded into the system.
3. The system reads each PDF, extracts the data, and compiles the results into a single Excel sheet.
4. Excel is ready for immediate statistical analysis.

## Contribution

Pull requests and issues are welcome. Please open an issue to discuss your proposed changes before submitting a PR.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

> Developed in collaboration with **DHsoft**  
> Empowering healthcare and research with intelligent automation.
