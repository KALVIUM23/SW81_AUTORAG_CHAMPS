from src.rag.llm_client import DiagnosticLLMClient

if __name__ == "__main__":
    client = DiagnosticLLMClient()
    
    mock_context = """
    Document: Apex Model X WSM - Section 7.3 (Rev 4.2)
    Model: Model X 2025 Hybrid (India BS-VI Stage 2)
    Procedure: For DTC P0420, inspect post-catalytic O2 sensor voltage at 2500 RPM.
    Acceptable fluctuation is <0.2V peak-to-peak. Torque specification for O2 sensor is 45 Nm.
    """

    print("Executing completion test...")
    response, telemetry = client.run_completion(
        vehicle_model="Model X",
        model_year=2025,
        variant="Hybrid",
        region="India",
        query="How do I verify DTC P0420 and what is the sensor torque?",
        retrieved_context=mock_context
    )

    print("\n--- STRUCTURED RESPONSE ---")
    print(f"Vehicle: {response.vehicle_context}")
    print(f"Summary: {response.summary}")
    for step in response.steps:
        print(f"Step {step.step_number}: {step.title} | Torque: {step.torque_or_electrical_spec}")
    print(f"\nTelemetry: {telemetry}")