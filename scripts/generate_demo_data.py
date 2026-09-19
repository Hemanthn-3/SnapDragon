"""
NEXUS Phase 15: Deterministic Synthetic Demo Dataset Generator
Creates 5 synthetic industrial inspection assets in demo_data/ with known relationships
and deliberately detectable engineering inconsistencies:
1. demo_data/reference_manual.txt (Technical reference baseline)
2. demo_data/inspection_report.pdf (Field inspection report with 245 Bar exceedance)
3. demo_data/scanned_maintenance_log.png (OCR document with 95 Nm undertorqued bolts)
4. demo_data/bearing_assembly_inspection.png (Bearing housing visual wear diagram)
5. demo_data/sensor_telemetry.json (SCADA structured telemetry with 5.8 mm/s vibration)
"""

import io
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

DEMO_DIR = Path("demo_data")
DEMO_DIR.mkdir(parents=True, exist_ok=True)


def generate_reference_manual():
    content = """================================================================================
TECHNICAL SPECIFICATION REFERENCE MANUAL
EQUIPMENT: Model TR-900 Heavy Industrial Gas Turbine / Bearing Housing B-42
DOCUMENT: SPEC-TR900-REV-4.2
CLASSIFICATION: Restricted Engineering Standard
================================================================================

1. HYDRAULIC PRESSURE OPERATING ENVELOPE
   - Nominal Operating Pressure: 180.0 to 210.0 Bar
   - Maximum Allowable Working Pressure (MAWP): 220.0 Bar
   - Critical Safety Cutoff: Systems operating above 220.0 Bar present severe seal
     failure risks and violate ASME Section VIII Div 1 safety codes.

2. FASTENER TORQUE SPECIFICATIONS
   - Fastener Specification: Grade 10.9 M16 Flange Retention Bolts
   - Required Assembly Torque: 125.0 Nm (+/- 5.0 Nm)
   - Minimum Safe Torque: 120.0 Nm
   - CRITICAL SAFETY HAZARD: Bolt torque below 120.0 Nm risks fastener fatigue,
     flange separation, and catastrophic bearing misalignment.

3. VIBRATION THRESHOLDS
   - Continuous Monitoring Channels: VIB_CH01 (Axial) and VIB_CH02 (Radial)
   - Baseline Nominal Vibration: <= 3.5 mm/s RMS
   - Maximum Permissible Operating Limit: 4.2 mm/s RMS
   - Exceedance Protocol: Any sustained vibration exceeding 4.2 mm/s requires
     immediate rotor balance inspection and speed reduction.

4. LUBRICATION STANDARDS
   - Required Lubricant: ISO VG 46 Fully Synthetic Turbine Fluid
   - Prohibited Fluids: Mineral engine oils, automotive SAE grades, or uncertified blends.

5. HOUSING INTEGRITY STANDARDS
   - External Bearing Housing B-42 must remain pristine with zero pitting,
     micro-fracturing, or thermal erosion along the bearing saddle.
================================================================================
"""
    manual_path = DEMO_DIR / "reference_manual.txt"
    manual_path.write_text(content, encoding="utf-8")
    print(f"Generated: {manual_path} ({len(content)} bytes)")


def generate_inspection_report_pdf():
    pdf_path = DEMO_DIR / "inspection_report.pdf"
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    
    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 740, "FIELD INSPECTION REPORT - UNIT TR-900")
    c.setFont("Helvetica", 10)
    c.drawString(50, 725, "Inspection ID: FIR-2026-0915 | Location: Sector 4 Energy Facility | Date: 2026-09-15")
    c.setLineWidth(1)
    c.line(50, 715, 560, 715)
    
    # Section 1: Executive Summary
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 690, "1. Executive Summary & Operating Parameters")
    c.setFont("Helvetica", 10)
    c.drawString(50, 670, "Routine quarterly inspection performed on Turbine Generator Set TR-900 while running.")
    c.drawString(50, 655, "Operational telemetry was collected under heavy base-load conditions.")
    
    # Section 2: Measured Parameters Table
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 625, "2. Recorded Engineering Measurements")
    c.setFont("Helvetica-Bold", 10)
    c.drawString(60, 605, "Parameter")
    c.drawString(220, 605, "Measured Value")
    c.drawString(340, 605, "Technician Note")
    c.line(50, 600, 560, 600)
    
    c.setFont("Helvetica", 10)
    c.drawString(60, 580, "Rotor Operating RPM")
    c.drawString(220, 580, "3,600 RPM")
    c.drawString(340, 580, "Synchronous 60 Hz")
    
    c.drawString(60, 560, "Hydraulic Fluid Pressure")
    c.setFont("Helvetica-Bold", 10)
    c.drawString(220, 560, "245.0 Bar")  # INCONSISTENCY: 245 > 220 Bar
    c.setFont("Helvetica", 10)
    c.drawString(340, 560, "Pressure high due to line regulator setting")
    
    c.drawString(60, 540, "Lubricant In Use")
    c.drawString(220, 540, "Mineral SAE 30")  # INCONSISTENCY: mineral instead of ISO VG 46
    c.drawString(340, 540, "Refilled by maintenance contractor")
    
    c.drawString(60, 520, "Bearing Housing Condition")
    c.drawString(220, 520, "Pristine (No Defect)")  # Contradicts image
    c.drawString(340, 520, "Visual check by inspector")

    c.line(50, 505, 560, 505)
    
    # Section 3: Recommendations
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 480, "3. Inspector Conclusion & Follow-up Actions")
    c.setFont("Helvetica", 10)
    c.drawString(50, 460, "The technician notes elevated pressure readings and recommended a comprehensive engineering")
    c.drawString(50, 445, "comparison against technical baseline specifications before certifying next quarterly cycle.")
    
    c.save()
    pdf_bytes = buf.getvalue()
    pdf_path.write_bytes(pdf_bytes)
    print(f"Generated: {pdf_path} ({len(pdf_bytes)} bytes)")


def generate_scanned_maintenance_log():
    """Generates synthetic scanned maintenance document with OCR-readable undertorqued bolt text."""
    img_path = DEMO_DIR / "scanned_maintenance_log.png"
    width, height = 800, 500
    img = Image.new("RGB", (width, height), color=(245, 245, 240))  # Slightly aged paper tone
    draw = ImageDraw.Draw(img)

    # Draw simulated borders
    draw.rectangle([(20, 20), (780, 480)], outline=(80, 80, 80), width=2)
    draw.rectangle([(24, 24), (776, 70)], fill=(220, 225, 230), outline=(100, 100, 100), width=1)

    # Header text
    draw.text((40, 35), "MAINTENANCE SERVICE ORDER & FASTENER LOG", fill=(20, 30, 50))
    draw.text((40, 85), "Facility: Station 4 Power Plant | Equipment: Turbine Unit TR-900", fill=(40, 40, 40))
    draw.text((40, 110), "Service Date: 2026-09-12 | Lead Mechanic: J. Vance | Shift: Alpha", fill=(40, 40, 40))
    draw.line([(30, 140), (770, 140)], fill=(120, 120, 120), width=1)

    # Key logged actions
    draw.text((40, 160), "WORK PERFORMED SUMMARY:", fill=(20, 20, 20))
    draw.text((50, 190), "1. Disassembled bearing saddle casing and inspected shaft seals.", fill=(30, 30, 30))
    draw.text((50, 220), "2. Cleaned mounting flange and reseated housing retention bolts.", fill=(30, 30, 30))
    
    # CRITICAL INCONSISTENCY LINE: 95 Nm (Required: 125 Nm)
    draw.text((50, 255), "3. TORQUE VERIFICATION: Applied Fastener Bolt Torque: 95 Nm", fill=(10, 10, 10))
    draw.text((50, 290), "4. Flange gap clearance: 0.18 mm measured with feeler gauge.", fill=(30, 30, 30))
    draw.text((50, 320), "5. Work completed per service bulletin instructions.", fill=(30, 30, 30))

    # Stamp / Signoff
    draw.rectangle([(520, 370), (750, 450)], outline=(180, 40, 40), width=2)
    draw.text((535, 385), "INSPECTION SIGN-OFF", fill=(180, 40, 40))
    draw.text((535, 410), "Status: LOGGED", fill=(180, 40, 40))
    draw.text((535, 425), "Torque Recorded: 95 Nm", fill=(180, 40, 40))

    img.save(img_path, format="PNG")
    print(f"Generated: {img_path} ({width}x{height})")


def generate_bearing_inspection_image():
    """Generates synthetic bearing housing inspection photo showing surface pitting wear."""
    img_path = DEMO_DIR / "bearing_assembly_inspection.png"
    width, height = 640, 480
    img = Image.new("RGB", (width, height), color=(45, 50, 58))  # Dark industrial machinery metal
    draw = ImageDraw.Draw(img)

    # Draw outer circular bearing casing
    draw.ellipse([(120, 60), (520, 420)], outline=(160, 170, 185), width=12, fill=(60, 65, 75))
    # Inner shaft collar
    draw.ellipse([(220, 150), (420, 330)], outline=(200, 210, 220), width=8, fill=(35, 40, 48))
    # Center shaft
    draw.ellipse([(280, 200), (360, 280)], fill=(90, 95, 105), outline=(180, 180, 180), width=4)

    # Synthetic abrasion & surface pitting defects on the top saddle
    defect_pixels = [
        (300, 80), (305, 82), (310, 78), (315, 85), (320, 82),
        (295, 88), (302, 92), (308, 90), (325, 88), (330, 95),
        (280, 95), (285, 90), (340, 92), (345, 86), (312, 96),
    ]
    for px, py in defect_pixels:
        draw.ellipse([(px - 6, py - 4), (px + 6, py + 4)], fill=(195, 80, 50), outline=(220, 120, 80))

    # Annotation tags
    draw.text((30, 30), "CAM-04 INSPECTION FEED: Unit TR-900 Bearing Housing B-42", fill=(240, 240, 240))
    draw.line([(350, 90), (440, 70)], fill=(255, 100, 60), width=2)
    draw.text((445, 62), "DEFECT: Surface Pitting & Abrasion", fill=(255, 120, 80))

    img.save(img_path, format="PNG")
    print(f"Generated: {img_path} ({width}x{height})")


def generate_sensor_telemetry_json():
    """Generates structured telemetry JSON containing vibration exceedance of 5.8 mm/s."""
    json_path = DEMO_DIR / "sensor_telemetry.json"
    telemetry_data = {
        "equipment_id": "TR-900-STATION-4",
        "timestamp_utc": "2026-09-15T14:30:00Z",
        "sampling_rate_hz": 1000,
        "operating_state": "BASE_LOAD_RUNNING",
        "channels": [
            {
                "channel_id": "VIB_CH01",
                "description": "Shaft Axial Vibration",
                "measured_rms": 3.1,
                "unit": "mm/s",
                "threshold_max": 4.2,
                "status": "NOMINAL",
            },
            {
                "channel_id": "VIB_CH02",
                "description": "Shaft Radial Vibration",
                "measured_rms": 5.8,  # INCONSISTENCY: 5.8 > 4.2 mm/s
                "unit": "mm/s",
                "threshold_max": 4.2,
                "status": "EXCEEDANCE",
            },
            {
                "channel_id": "PRESS_HYD01",
                "description": "Main Hydraulic Line Pressure",
                "measured_rms": 245.0,  # INCONSISTENCY: 245 > 220 Bar
                "unit": "Bar",
                "threshold_max": 220.0,
                "status": "EXCEEDANCE",
            },
            {
                "channel_id": "TEMP_BRG01",
                "description": "Bearing Housing Temperature",
                "measured_rms": 68.4,
                "unit": "Celsius",
                "threshold_max": 85.0,
                "status": "NOMINAL",
            },
        ],
        "system_alarms": [
            "ALARM_VIB_RADIAL_HIGH: Channel VIB_CH02 measured 5.8 mm/s (Limit 4.2 mm/s)",
            "ALARM_HYD_PRESSURE_CRITICAL: Channel PRESS_HYD01 measured 245.0 Bar (MAWP 220.0 Bar)",
        ],
    }
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(telemetry_data, f, indent=2)
    print(f"Generated: {json_path}")


def main():
    print("Generating synthetic demo dataset for NEXUS Phase 15...")
    generate_reference_manual()
    generate_inspection_report_pdf()
    generate_scanned_maintenance_log()
    generate_bearing_inspection_image()
    generate_sensor_telemetry_json()
    print("Synthetic dataset successfully created in demo_data/ directory.")


if __name__ == "__main__":
    main()
