#!/usr/bin/env python3
"""
Lambda Configuration Validation Script for NICE CKS GraphRAG system.
TASK-044: Configure Lambda settings (memory, timeout, concurrency)

This script validates that all Lambda configuration is properly set up
and provides a summary of the current settings.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def validate_sst_config() -> Dict[str, Any]:
    """Validate SST configuration file exists and has expected structure."""
    sst_config_path = project_root / "sst.config.ts"
    
    if not sst_config_path.exists():
        return {"status": "error", "message": "sst.config.ts not found"}
    
    try:
        # Read the config file and check for key components
        with open(sst_config_path, 'r') as f:
            content = f.read()
        
        required_components = [
            "timeout:", "memorySize:", "reservedConcurrentExecutions:",
            "QUERY_TIMEOUT_SECONDS", "SYNC_TIMEOUT_SECONDS", "MAX_CONTEXT_TOKENS",
            "functions/query.handler", "functions/health.handler", "functions/sync.scheduled_handler"
        ]
        
        missing_components = []
        for component in required_components:
            if component not in content:
                missing_components.append(component)
        
        return {
            "status": "success" if not missing_components else "warning",
            "missing_components": missing_components,
            "config_size": len(content),
            "has_query_config": "POST /query" in content,
            "has_health_config": "GET /health" in content,
            "has_sync_config": "scheduled_handler" in content
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Error reading SST config: {str(e)}"}

def validate_lambda_settings() -> Dict[str, Any]:
    """Validate Lambda settings configuration."""
    try:
        from config.lambda_settings import LambdaSettings, get_lambda_settings
        
        settings = LambdaSettings()
        
        validation_results = {
            "status": "success",
            "settings": {
                "query_timeout_seconds": settings.query_timeout_seconds,
                "sync_timeout_seconds": settings.sync_timeout_seconds,
                "max_context_tokens": settings.max_context_tokens,
                "batch_size": settings.batch_size,
                "openai_model": settings.openai_model,
                "openai_temperature": settings.openai_temperature,
                "mongodb_db_name": settings.mongodb_db_name,
                "mongodb_graph_collection": settings.mongodb_graph_collection,
                "mongodb_vector_collection": settings.mongodb_vector_collection,
                "log_level": settings.log_level,
                "environment": settings.environment
            },
            "validations": []
        }
        
        # Validate timeout settings
        if settings.query_timeout_seconds >= 30:
            validation_results["validations"].append({
                "type": "warning",
                "message": f"Query timeout ({settings.query_timeout_seconds}s) should be < 30s for Lambda"
            })
        
        if settings.sync_timeout_seconds >= 300:
            validation_results["validations"].append({
                "type": "warning", 
                "message": f"Sync timeout ({settings.sync_timeout_seconds}s) should be < 300s for Lambda"
            })
        
        # Validate temperature settings
        if not (0.0 <= settings.openai_temperature <= 2.0):
            validation_results["validations"].append({
                "type": "error",
                "message": f"OpenAI temperature ({settings.openai_temperature}) must be between 0.0 and 2.0"
            })
        
        # Validate context tokens
        if settings.max_context_tokens > 8000:
            validation_results["validations"].append({
                "type": "warning",
                "message": f"Max context tokens ({settings.max_context_tokens}) is very high, may cause high costs"
            })
        
        return validation_results
        
    except Exception as e:
        return {"status": "error", "message": f"Error validating Lambda settings: {str(e)}"}

def validate_performance_config() -> Dict[str, Any]:
    """Validate Lambda performance configuration."""
    perf_config_path = project_root / "config" / "lambda_performance.ts"
    
    if not perf_config_path.exists():
        return {"status": "error", "message": "lambda_performance.ts not found"}
    
    try:
        with open(perf_config_path, 'r') as f:
            content = f.read()
        
        required_configs = [
            "LAMBDA_PERFORMANCE_CONFIGS", "query:", "health:", "sync:",
            "memory:", "timeout:", "concurrency:", "COST_OPTIMIZATION_GUIDELINES",
            "MONITORING_METRICS"
        ]
        
        missing_configs = [config for config in required_configs if config not in content]
        
        return {
            "status": "success" if not missing_configs else "warning",
            "missing_configs": missing_configs,
            "config_size": len(content),
            "has_performance_configs": "LAMBDA_PERFORMANCE_CONFIGS" in content,
            "has_monitoring_metrics": "MONITORING_METRICS" in content,
            "has_cost_guidelines": "COST_OPTIMIZATION_GUIDELINES" in content
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Error reading performance config: {str(e)}"}

def validate_monitoring_script() -> Dict[str, Any]:
    """Validate Lambda performance monitoring script."""
    monitor_script_path = project_root / "scripts" / "lambda_performance_monitor.py"
    
    if not monitor_script_path.exists():
        return {"status": "error", "message": "lambda_performance_monitor.py not found"}
    
    try:
        with open(monitor_script_path, 'r') as f:
            content = f.read()
        
        required_classes = [
            "class LambdaMetrics:", "class PerformanceRecommendation:",
            "class LambdaPerformanceMonitor:", "def get_function_metrics",
            "def analyze_performance", "def generate_report"
        ]
        
        missing_classes = [cls for cls in required_classes if cls not in content]
        
        return {
            "status": "success" if not missing_classes else "warning",
            "missing_classes": missing_classes,
            "script_size": len(content),
            "is_executable": os.access(monitor_script_path, os.X_OK)
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Error reading monitoring script: {str(e)}"}

def validate_documentation() -> Dict[str, Any]:
    """Validate Lambda deployment documentation."""
    docs = [
        "docs/lambda-deployment-guide.md",
        "config/lambda_performance.ts"
    ]
    
    results = {"status": "success", "docs": {}}
    
    for doc_path in docs:
        full_path = project_root / doc_path
        if full_path.exists():
            with open(full_path, 'r') as f:
                content = f.read()
            results["docs"][doc_path] = {
                "exists": True,
                "size": len(content),
                "has_content": len(content) > 1000
            }
        else:
            results["docs"][doc_path] = {"exists": False}
            results["status"] = "warning"
    
    return results

def validate_lambda_functions() -> Dict[str, Any]:
    """Validate Lambda function implementations."""
    functions = [
        "functions/query.py",
        "functions/health.py", 
        "functions/sync.py"
    ]
    
    results = {"status": "success", "functions": {}}
    
    for func_path in functions:
        full_path = project_root / func_path
        if full_path.exists():
            with open(full_path, 'r') as f:
                content = f.read()
            
            results["functions"][func_path] = {
                "exists": True,
                "size": len(content),
                "has_mangum": "Mangum" in content,
                "has_fastapi": "FastAPI" in content,
                "has_lambda_settings": "lambda_settings" in content or "get_lambda_settings" in content
            }
        else:
            results["functions"][func_path] = {"exists": False}
            results["status"] = "warning"
    
    return results

def generate_summary_report() -> Dict[str, Any]:
    """Generate comprehensive validation report."""
    print("🔍 Validating Lambda Configuration for NICE CKS GraphRAG...")
    print("=" * 60)
    
    report = {
        "timestamp": __import__('datetime').datetime.utcnow().isoformat(),
        "validation_results": {}
    }
    
    validations = [
        ("SST Configuration", validate_sst_config),
        ("Lambda Settings", validate_lambda_settings),
        ("Performance Config", validate_performance_config),
        ("Monitoring Script", validate_monitoring_script),
        ("Documentation", validate_documentation),
        ("Lambda Functions", validate_lambda_functions)
    ]
    
    all_passed = True
    
    for name, validator in validations:
        print(f"\n📋 {name}:")
        try:
            result = validator()
            report["validation_results"][name.lower().replace(" ", "_")] = result
            
            if result["status"] == "success":
                print(f"  ✅ PASSED")
            elif result["status"] == "warning":
                print(f"  ⚠️  WARNING")
                all_passed = False
            else:
                print(f"  ❌ FAILED")
                all_passed = False
            
            # Print specific details
            if "message" in result:
                print(f"     {result['message']}")
            if "missing_components" in result and result["missing_components"]:
                print(f"     Missing: {', '.join(result['missing_components'])}")
            if "missing_configs" in result and result["missing_configs"]:
                print(f"     Missing: {', '.join(result['missing_configs'])}")
            if "missing_classes" in result and result["missing_classes"]:
                print(f"     Missing: {', '.join(result['missing_classes'])}")
            if "validations" in result:
                for validation in result["validations"]:
                    icon = "⚠️" if validation["type"] == "warning" else "❌"
                    print(f"     {icon} {validation['message']}")
                    
        except Exception as e:
            print(f"  ❌ ERROR: {str(e)}")
            report["validation_results"][name.lower().replace(" ", "_")] = {
                "status": "error", 
                "message": str(e)
            }
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All Lambda configurations are valid!")
        report["overall_status"] = "success"
    else:
        print("⚠️  Some configurations need attention")
        report["overall_status"] = "warning"
    
    return report

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate Lambda configuration")
    parser.add_argument("--output", type=str, help="Output file for JSON report")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed output")
    
    args = parser.parse_args()
    
    if args.quiet:
        # Redirect stdout temporarily
        import io
        import contextlib
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            report = generate_summary_report()
    else:
        report = generate_summary_report()
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Report saved to {args.output}")
    
    # Exit with appropriate code
    sys.exit(0 if report["overall_status"] == "success" else 1)

if __name__ == "__main__":
    main()