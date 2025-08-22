# Medical CRF OCR Quality Evaluation System

This project implements a **question-based OCR evaluation pipeline** specifically designed for medical Clinical Research Form (CRF) documents. Unlike traditional document-wide accuracy metrics, this system evaluates OCR quality at the individual question level, providing more precise reliability measures for clinical data collection workflows.

## Project Overview

<img width="196" height="51" alt="image" src="https://github.com/user-attachments/assets/8c531d07-2447-4c70-aa94-adef522e1f5c" />


- **Project Name**: Medical CRF Question-Based OCR Evaluator
- **Research Focus**: Layout-aware OCR evaluation for structured medical forms
- **Objective**: To provide question-level reliability metrics that directly support clinical data loading and quality assurance processes, moving beyond traditional full-text comparison approaches.

## Key Features

- 📋 **Question-Level Analysis**: Evaluates OCR accuracy at the actual data collection unit (individual questions/fields)
- 🔍 **Layout-Aware Processing**: Leverages PDF coordinate information to handle tables, checkboxes, and form structures
- 📊 **Multi-Dimensional Metrics**: Provides CER, WER, accuracy, string similarity, and GPT-based semantic confidence scores
- 🏥 **Medical Specialization**: Handles CRF-specific categories like disease history, family history, demographics
- 💰 **Budget Management**: Built-in OpenAI API usage monitoring and cost controls
- 📈 **Comprehensive Reporting**: Page-wise, category-wise, and question-type statistics with failure analysis

## System Architecture

```
[ Reference PDF (Template) ]
     ↓ (Layout-aware extraction)
[ Question Extractor (GPT + Rule-based) ]
     ↓
[ Scanned PDF ] → [ Naver Clova OCR ] → [ Question Matcher ]
     ↓                                        ↓
[ Enhanced Scoring Engine ] ← [ Multi-metric Calculator ]
     ↓
[ Statistical Analysis & Reporting ]
     ↓
[ JSON/CSV/Excel Results ]
```

## Installation

### Prerequisites
- Python 3.8+
- OpenAI API Key (for GPT-based question extraction)
- Naver Clova OCR API credentials
- Required Python packages (see requirements.txt)

### Installation Steps
```bash
git clone https://github.com/your-repo/medical-crf-ocr-evaluator.git
cd medical-crf-ocr-evaluator
pip install -r requirements.txt
```

### Environment Configuration
Create a `.env` file in the project root:
```bash
# OpenAI API
OPENAI_API_KEY=your_openai_api_key
OPENAI_GPT_MODEL=gpt-4o-mini

# Naver Clova OCR
NAVER_OCR_API_URL=your_naver_ocr_url
NAVER_OCR_SECRET_KEY=your_naver_secret_key

# Budget Controls
MONTHLY_BUDGET_USD=30.0
DAILY_BUDGET_USD=1.0
ENABLE_GPT_CONFIDENCE=true
GPT_CONFIDENCE_SAMPLE_RATE=0.2

# Processing Limits
MAX_PAGES_PER_RUN=10000
MAX_QUESTIONS_PER_RUN=1000000
ENABLE_LAYOUT_AWARE=true
```

### Running the Evaluation
```bash
# Run question-based CRF evaluation
python question_based_ocr_evaluator.py

# Or use the main orchestrator
python -m src.question_based_ocr_evaluator
```

## Tech Stack

- **OCR Engine**: Naver Clova OCR V2
- **Document Processing**: PyMuPDF (fitz), pdfplumber for layout analysis
- **Question Extraction**: OpenAI GPT-4 + rule-based fallback
- **Matching Algorithm**: Hungarian algorithm with multi-dimensional scoring
- **Text Metrics**: Custom CER/WER/Accuracy + RapidFuzz string similarity
- **Semantic Analysis**: GPT-based confidence scoring (optional)
- **Data Processing**: pandas, numpy for statistical analysis
- **Budget Management**: Custom cost tracking and API usage limits

## Core Components

### 1. Question Extractors (`src/extractors/`)
- **MedicalCRFQuestionExtractor**: GPT-powered + rule-based question identification
- Layout-aware processing for reference PDFs
- OCR block processing for scanned documents

### 2. OCR Client (`src/ocr/`)
- **NaverOCRClient**: Naver Clova OCR V2 integration
- Coordinate-aware text extraction
- Confidence score aggregation

### 3. Enhanced Matcher (`src/matchers/`)
- **EnhancedQuestionMatcher**: Hungarian algorithm-based optimal matching
- Multi-dimensional scoring (string similarity + format signals + domain keywords)
- Dynamic thresholds based on question types

### 4. Evaluation Pipeline (`question_based_ocr_evaluator.py`)
- End-to-end orchestration
- Statistical analysis and reporting
- CSV/JSON export with detailed breakdowns

## Example Use Case

1. **Input**: Reference CRF template (PDF) + Scanned completed forms (PDF)
2. **Processing**: 
   - Extract questions from reference using layout analysis
   - OCR scanned forms and extract questions
   - Match questions using enhanced scoring algorithm
   - Calculate multi-dimensional accuracy metrics
3. **Output**: 
   - Overall matching rate (e.g., 95.2%)
   - Category-wise statistics (demographics, disease history, etc.)
   - Failed item analysis with tagged failure reasons
   - Exportable reports for clinical QA workflows

## Evaluation Results Example

Based on WSCH Standard CRF (Ver. 3.0) testing:
- **Total Questions**: 21
- **Matching Rate**: 95.2% (20/21 matched)
- **Average String Similarity**: 0.822
- **Average CER**: 0.111
- **Average WER**: 0.237
- **Average Accuracy**: 0.723
- **Average Semantic Confidence**: 0.900

## File Structure

```
medical-crf-ocr-evaluator/
├── src/
│   ├── extractors/
│   │   └── medical_crf_extractor.py
│   ├── matchers/
│   │   └── enhanced_matcher.py
│   ├── ocr/
│   │   └── naver_clova_client.py
│   └── utils/
│       ├── accuracy_calculator.py
│       ├── budget_manager.py
│       ├── pdf_processor.py
│       └── text_normalizer.py
├── question_based_ocr_evaluator.py
├── improved_ocr_system.py (legacy)
├── data/input/
├── crf_evaluation_results/
└── requirements.txt
```

## Contribution

Pull requests and issues are welcome. Please open an issue to discuss proposed changes before submitting a PR. This system is designed for medical research environments and requires careful validation.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

> **Research Focus**: Question-based OCR evaluation for medical CRF documents  
> Advancing healthcare digitization with precision-driven quality metrics.
