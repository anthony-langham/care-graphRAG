#!/usr/bin/env python3
"""
Integration test runner for Care-GraphRAG system.
TASK-031: Comprehensive integration test execution with reporting.

Runs all integration tests in sequence, collects results, generates reports,
and provides CI/CD-ready output with proper exit codes.
"""

import os
import sys
import subprocess
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.logging import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)


class IntegrationTestRunner:
    """Integration test runner with comprehensive reporting."""
    
    def __init__(self, test_dir: Optional[Path] = None, output_dir: Optional[Path] = None):
        """
        Initialize integration test runner.
        
        Args:
            test_dir: Directory containing integration tests
            output_dir: Directory for test output and reports
        """
        self.project_root = project_root
        self.test_dir = test_dir or self.project_root / "tests" / "integration"
        self.output_dir = output_dir or self.project_root / "test_results"
        
        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True)
        
        # Test suite definitions
        self.test_suites = {
            "end_to_end": {
                "file": "test_end_to_end.py",
                "description": "End-to-end workflow tests",
                "timeout": 300,  # 5 minutes
                "critical": True
            },
            "error_scenarios": {
                "file": "test_error_scenarios.py", 
                "description": "Error handling and recovery tests",
                "timeout": 180,  # 3 minutes
                "critical": True
            },
            "load_performance": {
                "file": "test_load_performance.py",
                "description": "Load testing and performance benchmarks",
                "timeout": 600,  # 10 minutes
                "critical": False
            },
            "cost_tracking": {
                "file": "test_cost_tracking.py",
                "description": "Cost tracking and monitoring tests",
                "timeout": 120,  # 2 minutes
                "critical": False
            }
        }
        
        # Test results storage
        self.results = {}
        self.start_time = None
        self.end_time = None

    def run_all_tests(self, 
                      suites: Optional[List[str]] = None,
                      verbose: bool = True,
                      fail_fast: bool = False,
                      generate_report: bool = True) -> bool:
        """
        Run all integration test suites.
        
        Args:
            suites: Specific test suites to run (None for all)
            verbose: Enable verbose output
            fail_fast: Stop on first failure
            generate_report: Generate comprehensive report
            
        Returns:
            True if all tests passed, False otherwise
        """
        logger.info("Starting integration test execution")
        self.start_time = datetime.now()
        
        # Determine which suites to run
        suites_to_run = suites or list(self.test_suites.keys())
        
        logger.info(f"Running test suites: {', '.join(suites_to_run)}")
        
        overall_success = True
        
        for suite_name in suites_to_run:
            if suite_name not in self.test_suites:
                logger.error(f"Unknown test suite: {suite_name}")
                overall_success = False
                continue
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Running {suite_name} test suite")
            logger.info(f"{'='*60}")
            
            suite_success = self._run_test_suite(suite_name, verbose)
            
            if not suite_success:
                overall_success = False
                
                # Check if this is a critical suite and fail_fast is enabled
                if self.test_suites[suite_name]["critical"] and fail_fast:
                    logger.error(f"Critical test suite {suite_name} failed, stopping execution")
                    break
        
        self.end_time = datetime.now()
        
        # Generate comprehensive report
        if generate_report:
            self._generate_integration_report()
        
        # Log summary
        duration = self.end_time - self.start_time
        logger.info(f"\n{'='*60}")
        logger.info(f"Integration test execution completed in {duration}")
        logger.info(f"Overall result: {'PASSED' if overall_success else 'FAILED'}")
        logger.info(f"{'='*60}")
        
        return overall_success

    def _run_test_suite(self, suite_name: str, verbose: bool = True) -> bool:
        """
        Run a specific test suite.
        
        Args:
            suite_name: Name of the test suite
            verbose: Enable verbose output
            
        Returns:
            True if suite passed, False otherwise
        """
        suite_config = self.test_suites[suite_name]
        test_file = self.test_dir / suite_config["file"]
        
        if not test_file.exists():
            logger.error(f"Test file not found: {test_file}")
            self.results[suite_name] = {
                "status": "ERROR",
                "error": "Test file not found",
                "duration": 0,
                "tests_run": 0,
                "failures": 0,
                "errors": 1
            }
            return False
        
        # Prepare pytest command
        junit_output = self.output_dir / f"integration_{suite_name}.xml"
        json_output = self.output_dir / f"integration_{suite_name}.json"
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(test_file),
            f"--junit-xml={junit_output}",
            f"--json-report", f"--json-report-file={json_output}",
            "--tb=short",
            "--durations=10"
        ]
        
        if verbose:
            cmd.append("-v")
        
        # Add timeout
        timeout = suite_config.get("timeout", 300)
        
        logger.info(f"Running command: {' '.join(cmd)}")
        logger.info(f"Timeout: {timeout} seconds")
        
        # Execute test suite
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            duration = time.time() - start_time
            
            # Parse results
            suite_results = self._parse_test_results(
                suite_name, result, duration, junit_output, json_output
            )
            
            self.results[suite_name] = suite_results
            
            # Log results
            logger.info(f"\nTest suite {suite_name} completed:")
            logger.info(f"  Status: {suite_results['status']}")
            logger.info(f"  Duration: {suite_results['duration']:.2f} seconds")
            logger.info(f"  Tests run: {suite_results['tests_run']}")
            logger.info(f"  Failures: {suite_results['failures']}")
            logger.info(f"  Errors: {suite_results['errors']}")
            
            if verbose and result.stdout:
                logger.info(f"\nTest output:\n{result.stdout}")
            
            if result.stderr:
                logger.warning(f"\nTest stderr:\n{result.stderr}")
            
            return suite_results["status"] == "PASSED"
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            logger.error(f"Test suite {suite_name} timed out after {timeout} seconds")
            
            self.results[suite_name] = {
                "status": "TIMEOUT",
                "error": f"Timed out after {timeout} seconds",
                "duration": duration,
                "tests_run": 0,
                "failures": 0,
                "errors": 1
            }
            
            return False
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Failed to run test suite {suite_name}: {e}")
            
            self.results[suite_name] = {
                "status": "ERROR",
                "error": str(e),
                "duration": duration,
                "tests_run": 0,
                "failures": 0,
                "errors": 1
            }
            
            return False

    def _parse_test_results(self, 
                           suite_name: str, 
                           result: subprocess.CompletedProcess,
                           duration: float,
                           junit_file: Path,
                           json_file: Path) -> Dict[str, Any]:
        """
        Parse test results from pytest output.
        
        Args:
            suite_name: Name of test suite
            result: Subprocess result
            duration: Test execution duration
            junit_file: JUnit XML output file
            json_file: JSON report output file
            
        Returns:
            Dictionary with parsed test results
        """
        parsed_results = {
            "status": "UNKNOWN",
            "duration": duration,
            "tests_run": 0,
            "failures": 0,
            "errors": 0,
            "exit_code": result.returncode,
            "output": result.stdout,
            "stderr": result.stderr
        }
        
        # Parse JUnit XML if available
        if junit_file.exists():
            try:
                tree = ET.parse(junit_file)
                root = tree.getroot()
                
                parsed_results.update({
                    "tests_run": int(root.get("tests", 0)),
                    "failures": int(root.get("failures", 0)),
                    "errors": int(root.get("errors", 0)),
                    "duration": float(root.get("time", duration))
                })
                
            except Exception as e:
                logger.warning(f"Failed to parse JUnit XML for {suite_name}: {e}")
        
        # Parse JSON report if available
        if json_file.exists():
            try:
                with open(json_file, 'r') as f:
                    json_data = json.load(f)
                
                summary = json_data.get("summary", {})
                parsed_results.update({
                    "tests_run": summary.get("total", 0),
                    "failures": summary.get("failed", 0),
                    "errors": summary.get("error", 0),
                    "passed": summary.get("passed", 0),
                    "skipped": summary.get("skipped", 0)
                })
                
            except Exception as e:
                logger.warning(f"Failed to parse JSON report for {suite_name}: {e}")
        
        # Determine status
        if result.returncode == 0:
            parsed_results["status"] = "PASSED"
        elif parsed_results["failures"] > 0 or parsed_results["errors"] > 0:
            parsed_results["status"] = "FAILED"
        else:
            parsed_results["status"] = "ERROR"
        
        return parsed_results

    def _generate_integration_report(self):
        """Generate comprehensive integration test report."""
        logger.info("Generating integration test report")
        
        # Calculate summary statistics
        total_duration = (self.end_time - self.start_time).total_seconds()
        total_tests = sum(r.get("tests_run", 0) for r in self.results.values())
        total_failures = sum(r.get("failures", 0) for r in self.results.values())
        total_errors = sum(r.get("errors", 0) for r in self.results.values())
        total_passed = sum(r.get("passed", 0) for r in self.results.values())
        
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        # Generate report data
        report_data = {
            "execution_info": {
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat(),
                "total_duration": total_duration,
                "test_runner_version": "1.0.0"
            },
            "summary": {
                "total_tests": total_tests,
                "passed": total_passed,
                "failures": total_failures,
                "errors": total_errors,
                "success_rate": success_rate,
                "overall_status": "PASSED" if total_failures == 0 and total_errors == 0 else "FAILED"
            },
            "suites": self.results,
            "system_info": {
                "python_version": sys.version,
                "platform": sys.platform,
                "working_directory": str(self.project_root)
            }
        }
        
        # Write JSON report
        json_report_file = self.output_dir / "integration_test_report.json"
        with open(json_report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Write HTML report if possible
        try:
            self._generate_html_report(report_data)
        except Exception as e:
            logger.warning(f"Failed to generate HTML report: {e}")
        
        # Write summary text report
        self._generate_text_report(report_data)
        
        logger.info(f"Integration test report generated: {json_report_file}")

    def _generate_html_report(self, report_data: Dict[str, Any]):
        """Generate HTML test report."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Integration Test Report - Care-GraphRAG</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ background-color: #e8f5e8; padding: 15px; margin: 20px 0; border-radius: 5px; }}
        .failed {{ background-color: #ffe8e8; }}
        .suite {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .passed {{ border-left: 5px solid #4CAF50; }}
        .failed-suite {{ border-left: 5px solid #f44336; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Integration Test Report</h1>
        <p>Care-GraphRAG System Integration Tests</p>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="summary {'failed' if report_data['summary']['overall_status'] == 'FAILED' else ''}">
        <h2>Summary</h2>
        <table>
            <tr><td><strong>Overall Status:</strong></td><td>{report_data['summary']['overall_status']}</td></tr>
            <tr><td><strong>Total Tests:</strong></td><td>{report_data['summary']['total_tests']}</td></tr>
            <tr><td><strong>Passed:</strong></td><td>{report_data['summary']['passed']}</td></tr>
            <tr><td><strong>Failed:</strong></td><td>{report_data['summary']['failures']}</td></tr>
            <tr><td><strong>Errors:</strong></td><td>{report_data['summary']['errors']}</td></tr>
            <tr><td><strong>Success Rate:</strong></td><td>{report_data['summary']['success_rate']:.1f}%</td></tr>
            <tr><td><strong>Total Duration:</strong></td><td>{report_data['execution_info']['total_duration']:.1f} seconds</td></tr>
        </table>
    </div>
    
    <h2>Test Suites</h2>
"""
        
        for suite_name, suite_results in report_data['suites'].items():
            status_class = "passed" if suite_results['status'] == 'PASSED' else "failed-suite"
            suite_config = self.test_suites.get(suite_name, {})
            
            html_content += f"""
    <div class="suite {status_class}">
        <h3>{suite_name.replace('_', ' ').title()}</h3>
        <p>{suite_config.get('description', 'No description available')}</p>
        <table>
            <tr><td><strong>Status:</strong></td><td>{suite_results['status']}</td></tr>
            <tr><td><strong>Duration:</strong></td><td>{suite_results.get('duration', 0):.2f} seconds</td></tr>
            <tr><td><strong>Tests Run:</strong></td><td>{suite_results.get('tests_run', 0)}</td></tr>
            <tr><td><strong>Passed:</strong></td><td>{suite_results.get('passed', 0)}</td></tr>
            <tr><td><strong>Failures:</strong></td><td>{suite_results.get('failures', 0)}</td></tr>
            <tr><td><strong>Errors:</strong></td><td>{suite_results.get('errors', 0)}</td></tr>
        </table>
        {f'<p><strong>Error:</strong> {suite_results.get("error", "")}</p>' if suite_results.get('error') else ''}
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        
        html_report_file = self.output_dir / "integration_test_report.html"
        with open(html_report_file, 'w') as f:
            f.write(html_content)

    def _generate_text_report(self, report_data: Dict[str, Any]):
        """Generate text summary report."""
        text_content = f"""
INTEGRATION TEST REPORT
Care-GraphRAG System
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*60}
SUMMARY
{'='*60}
Overall Status: {report_data['summary']['overall_status']}
Total Tests: {report_data['summary']['total_tests']}
Passed: {report_data['summary']['passed']}
Failed: {report_data['summary']['failures']}
Errors: {report_data['summary']['errors']}
Success Rate: {report_data['summary']['success_rate']:.1f}%
Total Duration: {report_data['execution_info']['total_duration']:.1f} seconds

{'='*60}
TEST SUITES
{'='*60}
"""
        
        for suite_name, suite_results in report_data['suites'].items():
            suite_config = self.test_suites.get(suite_name, {})
            status_indicator = "✓" if suite_results['status'] == 'PASSED' else "✗"
            
            text_content += f"""
{status_indicator} {suite_name.replace('_', ' ').title()}
   Description: {suite_config.get('description', 'No description')}
   Status: {suite_results['status']}
   Duration: {suite_results.get('duration', 0):.2f}s
   Tests: {suite_results.get('tests_run', 0)} run, {suite_results.get('passed', 0)} passed, {suite_results.get('failures', 0)} failed, {suite_results.get('errors', 0)} errors
   {f'Error: {suite_results.get("error", "")}' if suite_results.get('error') else ''}

"""
        
        text_content += f"""
{'='*60}
EXECUTION INFO
{'='*60}
Start Time: {report_data['execution_info']['start_time']}
End Time: {report_data['execution_info']['end_time']}
Python Version: {report_data['system_info']['python_version']}
Platform: {report_data['system_info']['platform']}
Working Directory: {report_data['system_info']['working_directory']}
"""
        
        text_report_file = self.output_dir / "integration_test_summary.txt"
        with open(text_report_file, 'w') as f:
            f.write(text_content)


def main():
    """Main entry point for integration test runner."""
    parser = argparse.ArgumentParser(
        description="Run Care-GraphRAG integration tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_integration_tests.py                    # Run all tests
  python scripts/run_integration_tests.py --suites end_to_end error_scenarios  # Run specific suites
  python scripts/run_integration_tests.py --fail-fast       # Stop on first failure
  python scripts/run_integration_tests.py --quiet           # Minimal output
        """
    )
    
    parser.add_argument(
        "--suites", 
        nargs="*", 
        choices=["end_to_end", "error_scenarios", "load_performance", "cost_tracking"],
        help="Specific test suites to run (default: all)"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=Path,
        help="Directory for test output (default: test_results/)"
    )
    
    parser.add_argument(
        "--fail-fast", 
        action="store_true",
        help="Stop execution on first critical test failure"
    )
    
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Minimal output (opposite of verbose)"
    )
    
    parser.add_argument(
        "--no-report", 
        action="store_true",
        help="Skip generating comprehensive report"
    )
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = IntegrationTestRunner(output_dir=args.output_dir)
    
    # Run tests
    success = runner.run_all_tests(
        suites=args.suites,
        verbose=not args.quiet,
        fail_fast=args.fail_fast,
        generate_report=not args.no_report
    )
    
    # Exit with appropriate code for CI/CD
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()