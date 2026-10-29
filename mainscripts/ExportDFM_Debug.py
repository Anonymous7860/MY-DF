"""
Debug export wrapper for SAENXT model.
Calls export_dfm_debug() instead of export_dfm().
"""
from pathlib import Path
import models
from core.interact import interact as io


def main(model_class_name, saved_models_path):
    """Run debug DFM export."""
    io.log_info("="*70)
    io.log_info("SAENXT DEBUG DFM EXPORT")
    io.log_info("="*70)
    io.log_info("This will export with comprehensive debugging:")
    io.log_info("  - Verbose ONNX conversion logs")
    io.log_info("  - Graph validation and analysis")
    io.log_info("  - PyTorch vs ONNX output comparison")
    io.log_info("  - Precision drift metrics")
    io.log_info("  - Visual comparison images")
    io.log_info("="*70)
    io.log_info("")
    
    model = models.import_model(model_class_name)(
        is_exporting=True,
        saved_models_path=saved_models_path,
        cpu_only=True
    )
    
    # Call the debug export method instead of regular export
    model.export_dfm_debug()

