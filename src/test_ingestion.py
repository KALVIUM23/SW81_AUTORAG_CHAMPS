from pathlib import Path
from src.ingestion.cleaner import TextCleaner
from src.ingestion.chunker import AutomotiveChunker
from src.ingestion.validator import IngestionValidator

if __name__ == "__main__":
    sample_text = """
    Apex Motors Workshop Manual - Rev 4.2
    Confidential - Authorized Service Bay Only
    
    Section 7.3: Catalytic Converter Diagnostic & Threshold Verification
    Model: Model X | Year: 2025 | Variant: Hybrid (2.0L Atkinson) | Region: India (BS-VI Stage 2)

    Procedure for DTC P0420 (Catalyst System Efficiency Below Threshold - Bank 1):
    1. Inspect pre-catalyst and post-catalyst exhaust joints for physical fractures or soot.
    2. Warm engine to operating temperature (>85°C). Run at 2500 RPM for 3 minutes.
    3. Measure downstream O2 sensor voltage fluctuation. Acceptable variance is <0.2V peak-to-peak.
    4. Fastener torque requirement: Oxygen Sensor to exhaust manifold must be torqued to 45 Nm.
    """
    
    cleaned_text = TextCleaner.clean_page_text(sample_text)
    pages = [{"page_number": 37, "raw_text": cleaned_text}]
    
    metadata = {
        "document_id": "WSM-2025-MODX-IND-001",
        "document_name": "Apex Model X 2025 Powertrain Manual",
        "vehicle_model": "Model X",
        "model_year": 2025,
        "variant": "Hybrid",
        "region": "India",
        "document_type": "WSM",
        "version": "4.2",
        "status": "ACTIVE"
    }

    chunker = AutomotiveChunker(chunk_size_tokens=512, overlap_tokens=64)
    chunks = chunker.chunk_document(metadata, pages)
    report = IngestionValidator.generate_report(chunks)

    print("=== INGESTION INTEGRITY REPORT ===")
    for k, v in report.items():
        print(f"{k}: {v}")
        
    print(f"\nGenerated Chunk ID: {chunks[0]['chunk_id']}")
    print(f"Token Count: {chunks[0]['token_count']}")
    print(f"Cleaned Content:\n{chunks[0]['content']}")