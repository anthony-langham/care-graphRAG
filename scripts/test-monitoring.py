#!/usr/bin/env python3
"""
Test script for validating production monitoring setup.
Tests CloudWatch metrics, alarms, X-Ray tracing, and SNS notifications.
"""

import json
import time
import requests
import boto3
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MonitoringTester:
    """Test production monitoring setup for NICE CKS GraphRAG."""
    
    def __init__(self, region: str = "eu-west-2", stage: str = "production"):
        self.region = region
        self.stage = stage
        self.app_name = "nice-cks-graphrag"
        
        # Initialize AWS clients
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
        self.xray = boto3.client('xray', region_name=region)
        self.sns = boto3.client('sns', region_name=region)
        self.lambda_client = boto3.client('lambda', region_name=region)
        
        # API endpoint (should be provided or discovered)
        self.api_endpoint = None
        
    def discover_api_endpoint(self) -> Optional[str]:
        """Discover API Gateway endpoint from SST outputs."""
        try:
            # Try to get from SST outputs if available
            # In practice, this would be passed as parameter or environment variable
            logger.info("API endpoint discovery not implemented - should be provided")
            return None
        except Exception as e:
            logger.warning(f"Could not discover API endpoint: {e}")
            return None
    
    def test_cloudwatch_dashboard(self) -> Dict[str, any]:
        """Test CloudWatch dashboard exists and is accessible."""
        dashboard_name = f"{self.app_name}-{self.stage}"
        
        try:
            response = self.cloudwatch.describe_dashboards(
                DashboardNames=[dashboard_name]
            )
            
            if response['DashboardEntries']:
                dashboard = response['DashboardEntries'][0]
                logger.info(f"✅ Dashboard found: {dashboard_name}")
                return {
                    "status": "success",
                    "dashboard_name": dashboard_name,
                    "last_modified": dashboard.get('LastModified'),
                    "size": dashboard.get('Size')
                }
            else:
                logger.error(f"❌ Dashboard not found: {dashboard_name}")
                return {"status": "error", "message": "Dashboard not found"}
                
        except Exception as e:
            logger.error(f"❌ Error checking dashboard: {e}")
            return {"status": "error", "message": str(e)}
    
    def test_cloudwatch_alarms(self) -> Dict[str, any]:
        """Test CloudWatch alarms are configured correctly."""
        expected_alarms = [
            f"{self.app_name}-{self.stage}-query-errors",
            f"{self.app_name}-{self.stage}-query-duration", 
            f"{self.app_name}-{self.stage}-health-failures",
            f"{self.app_name}-{self.stage}-api-5xx-errors"
        ]
        
        results = {"status": "success", "alarms": {}}
        
        for alarm_name in expected_alarms:
            try:
                response = self.cloudwatch.describe_alarms(
                    AlarmNames=[alarm_name]
                )
                
                if response['MetricAlarms']:
                    alarm = response['MetricAlarms'][0]
                    results["alarms"][alarm_name] = {
                        "exists": True,
                        "state": alarm['StateValue'],
                        "threshold": alarm['Threshold'],
                        "comparison": alarm['ComparisonOperator']
                    }
                    logger.info(f"✅ Alarm configured: {alarm_name} ({alarm['StateValue']})")
                else:
                    results["alarms"][alarm_name] = {"exists": False}
                    results["status"] = "partial"
                    logger.warning(f"⚠️  Alarm not found: {alarm_name}")
                    
            except Exception as e:
                results["alarms"][alarm_name] = {"exists": False, "error": str(e)}
                results["status"] = "error"
                logger.error(f"❌ Error checking alarm {alarm_name}: {e}")
        
        return results
    
    def test_xray_tracing(self) -> Dict[str, any]:
        """Test X-Ray tracing is collecting data."""
        try:
            # Get service map for last hour
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=1)
            
            response = self.xray.get_service_graph(
                StartTime=start_time,
                EndTime=end_time
            )
            
            services = response.get('Services', [])
            
            if services:
                logger.info(f"✅ X-Ray collecting data from {len(services)} services")
                service_names = [s.get('Name', 'unknown') for s in services]
                return {
                    "status": "success",
                    "services_count": len(services),
                    "service_names": service_names
                }
            else:
                logger.warning("⚠️  No X-Ray trace data found (may be normal for new deployment)")
                return {
                    "status": "no_data",
                    "message": "No trace data available"
                }
                
        except Exception as e:
            logger.error(f"❌ Error checking X-Ray: {e}")
            return {"status": "error", "message": str(e)}
    
    def test_lambda_functions(self) -> Dict[str, any]:
        """Test Lambda functions are configured with monitoring."""
        function_patterns = [
            f"{self.app_name}-{self.stage}-query",
            f"{self.app_name}-{self.stage}-health"
        ]
        
        results = {"status": "success", "functions": {}}
        
        try:
            # List all functions and find matches
            paginator = self.lambda_client.get_paginator('list_functions')
            
            for page in paginator.paginate():
                for func in page['Functions']:
                    func_name = func['FunctionName']
                    
                    # Check if this matches our expected patterns
                    for pattern in function_patterns:
                        if pattern in func_name:
                            # Get detailed function configuration
                            config = self.lambda_client.get_function_configuration(
                                FunctionName=func_name
                            )
                            
                            results["functions"][func_name] = {
                                "runtime": config.get('Runtime'),
                                "timeout": config.get('Timeout'),
                                "memory": config.get('MemorySize'),
                                "tracing_enabled": config.get('TracingConfig', {}).get('Mode') == 'Active',
                                "last_modified": config.get('LastModified')
                            }
                            
                            logger.info(f"✅ Function found: {func_name}")
                            break
            
            if not results["functions"]:
                results["status"] = "error"
                logger.error("❌ No Lambda functions found matching expected patterns")
        
        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)
            logger.error(f"❌ Error checking Lambda functions: {e}")
        
        return results
    
    def test_api_health(self, api_endpoint: str) -> Dict[str, any]:
        """Test API health endpoint and measure response time."""
        if not api_endpoint:
            return {"status": "skipped", "message": "No API endpoint provided"}
        
        health_url = f"{api_endpoint}/health"
        
        try:
            start_time = time.time()
            response = requests.get(health_url, timeout=30)
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            if response.status_code == 200:
                health_data = response.json()
                logger.info(f"✅ Health check passed ({response_time:.0f}ms)")
                return {
                    "status": "success",
                    "response_time_ms": response_time,
                    "health_data": health_data
                }
            else:
                logger.error(f"❌ Health check failed: {response.status_code}")
                return {
                    "status": "error",
                    "status_code": response.status_code,
                    "response_time_ms": response_time
                }
                
        except Exception as e:
            logger.error(f"❌ Health check error: {e}")
            return {"status": "error", "message": str(e)}
    
    def test_metrics_collection(self) -> Dict[str, any]:
        """Test that CloudWatch metrics are being collected."""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=1)
            
            # Check for Lambda metrics
            response = self.cloudwatch.get_metric_statistics(
                Namespace='AWS/Lambda',
                MetricName='Invocations',
                Dimensions=[],
                StartTime=start_time,
                EndTime=end_time,
                Period=300,  # 5 minutes
                Statistics=['Sum']
            )
            
            datapoints = response.get('Datapoints', [])
            
            if datapoints:
                total_invocations = sum(dp['Sum'] for dp in datapoints)
                logger.info(f"✅ Metrics collection active ({total_invocations} invocations in last hour)")
                return {
                    "status": "success",
                    "datapoints": len(datapoints),
                    "total_invocations": total_invocations
                }
            else:
                logger.warning("⚠️  No metrics data found (may be normal for new deployment)")
                return {"status": "no_data", "message": "No metrics available"}
                
        except Exception as e:
            logger.error(f"❌ Error checking metrics: {e}")
            return {"status": "error", "message": str(e)}
    
    def run_comprehensive_test(self, api_endpoint: Optional[str] = None) -> Dict[str, any]:
        """Run all monitoring tests and return comprehensive results."""
        logger.info("🔍 Starting comprehensive monitoring test...")
        
        if api_endpoint:
            self.api_endpoint = api_endpoint
        else:
            self.api_endpoint = self.discover_api_endpoint()
        
        results = {
            "test_timestamp": datetime.utcnow().isoformat(),
            "region": self.region,
            "stage": self.stage,
            "app_name": self.app_name,
            "tests": {}
        }
        
        # Run all tests
        test_functions = [
            ("dashboard", self.test_cloudwatch_dashboard),
            ("alarms", self.test_cloudwatch_alarms),
            ("xray", self.test_xray_tracing),
            ("lambda_functions", self.test_lambda_functions),
            ("api_health", lambda: self.test_api_health(self.api_endpoint)),
            ("metrics", self.test_metrics_collection)
        ]
        
        for test_name, test_func in test_functions:
            logger.info(f"Running {test_name} test...")
            try:
                results["tests"][test_name] = test_func()
            except Exception as e:
                logger.error(f"Test {test_name} failed: {e}")
                results["tests"][test_name] = {"status": "error", "message": str(e)}
        
        # Calculate overall status
        statuses = [test["status"] for test in results["tests"].values()]
        if all(s == "success" for s in statuses):
            results["overall_status"] = "success"
        elif any(s == "error" for s in statuses):
            results["overall_status"] = "error"
        else:
            results["overall_status"] = "partial"
        
        logger.info(f"🏁 Comprehensive test completed: {results['overall_status']}")
        return results
    
    def generate_report(self, results: Dict[str, any]) -> str:
        """Generate a human-readable monitoring test report."""
        report = []
        report.append("=" * 60)
        report.append("NICE CKS GraphRAG - Monitoring Test Report")
        report.append("=" * 60)
        report.append(f"Test Time: {results['test_timestamp']}")
        report.append(f"Region: {results['region']}")
        report.append(f"Stage: {results['stage']}")
        report.append(f"Overall Status: {results['overall_status'].upper()}")
        report.append("")
        
        for test_name, test_result in results["tests"].items():
            status_emoji = {
                "success": "✅",
                "error": "❌", 
                "partial": "⚠️",
                "no_data": "ℹ️",
                "skipped": "⏭️"
            }.get(test_result["status"], "❓")
            
            report.append(f"{status_emoji} {test_name.replace('_', ' ').title()}: {test_result['status']}")
            
            if test_result["status"] == "error" and "message" in test_result:
                report.append(f"   Error: {test_result['message']}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


def main():
    """Main function to run monitoring tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test NICE CKS GraphRAG monitoring setup')
    parser.add_argument('--region', default='eu-west-2', help='AWS region')
    parser.add_argument('--stage', default='production', help='Deployment stage')
    parser.add_argument('--api-endpoint', help='API Gateway endpoint URL')
    parser.add_argument('--output', help='Output file for test results (JSON)')
    
    args = parser.parse_args()
    
    tester = MonitoringTester(region=args.region, stage=args.stage)
    results = tester.run_comprehensive_test(api_endpoint=args.api_endpoint)
    
    # Print report
    report = tester.generate_report(results)
    print(report)
    
    # Save detailed results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Detailed results saved to: {args.output}")
    
    # Exit with appropriate code
    if results["overall_status"] == "success":
        exit(0)
    elif results["overall_status"] == "partial":
        exit(1)
    else:
        exit(2)


if __name__ == "__main__":
    main()