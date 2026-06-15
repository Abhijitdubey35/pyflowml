# PyFlowML Scripts

Utility scripts for documentation generation and reporting.

## Available Scripts

### `generate_implementation_plan.py`

Generates a professional PDF document outlining the v1.0.4 → v2.0 implementation roadmap for PyFlowML.

**Features:**
- Professional multi-page layout with branded styling
- Executive summary of v1.0.4 audit findings
- Phase map showing 4 sequential release cycles
- 9 detailed upgrade cards with implementation details
- Release checklist and versioning strategy
- Dark theme with custom color palette

**Usage:**

```bash
python scripts/generate_implementation_plan.py
```

**Output:**
- Generated PDF: `./reports/PyFlowML_Implementation_Plan.pdf`

**Requirements:**
- `reportlab` (install with: `pip install reportlab`)

**Notes:**
- The script uses relative paths and will automatically create the `reports/` directory if it doesn't exist
- Output is a self-contained PDF ready for sharing with stakeholders
- The document is marked as "Confidential" and includes the current date
