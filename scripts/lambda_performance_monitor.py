#!/usr/bin/env python3
"""
Lambda Performance Monitor for NICE CKS GraphRAG system.
TASK-044: Configure Lambda settings (memory, timeout, concurrency)

This script monitors Lambda function performance and provides recommendations
for optimizing memory, timeout, and concurrency settings based on CloudWatch metrics.
"""

import json
import boto3
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class LambdaMetrics:
    """Lambda function performance metrics."""
    function_name: str
    avg_duration: float
    max_duration: float
    avg_memory_used: float
    max_memory_used: float
    memory_size: int
    error_count: int
    throttle_count: int
    invocation_count: int
    cost_estimate: float

@dataclass
class PerformanceRecommendation:
    """Performance optimization recommendation."""
    metric_type: str
    current_value: float
    recommended_value: float
    reasoning: str
    potential_savings: Optional[float] = None


class LambdaPerformanceMonitor:
    """Monitor and analyze Lambda function performance."""
    
    def __init__(self, region: str = "eu-west-2"):
        """
        Initialize the performance monitor.
        
        Args:
            region: AWS region for Lambda functions
        """
        self.region = region
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
        self.lambda_client = boto3.client('lambda', region_name=region)
        
        # Lambda pricing (EU-West-2 as of 2024)
        self.price_per_gb_second = 0.0000166667
        self.price_per_request = 0.0000002
    
    def get_function_metrics(self, function_name: str, hours: int = 24) -> LambdaMetrics:
        """
        Get performance metrics for a Lambda function.
        
        Args:
            function_name: Name of the Lambda function
            hours: Number of hours to look back for metrics
            
        Returns:
            LambdaMetrics: Performance metrics for the function
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        # Get function configuration
        try:
            function_config = self.lambda_client.get_function_configuration(
                FunctionName=function_name
            )
            memory_size = function_config['MemorySize']
        except Exception as e:
            print(f"Warning: Could not get function config for {function_name}: {e}")
            memory_size = 1024  # Default
        
        # Get metrics from CloudWatch
        metrics = {}
        
        # Duration metrics
        duration_response = self.cloudwatch.get_metric_statistics(
            Namespace='AWS/Lambda',
            MetricName='Duration',
            Dimensions=[{'Name': 'FunctionName', 'Value': function_name}],
            StartTime=start_time,
            EndTime=end_time,
            Period=3600,  # 1 hour periods
            Statistics=['Average', 'Maximum']
        )
        
        # Memory utilization (custom metric if available)
        memory_response = self.cloudwatch.get_metric_statistics(
            Namespace='AWS/Lambda',
            MetricName='MemoryUtilization',
            Dimensions=[{'Name': 'FunctionName', 'Value': function_name}],
            StartTime=start_time,
            EndTime=end_time,
            Period=3600,
            Statistics=['Average', 'Maximum']
        )
        
        # Error count
        error_response = self.cloudwatch.get_metric_statistics(
            Namespace='AWS/Lambda',
            MetricName='Errors',
            Dimensions=[{'Name': 'FunctionName', 'Value': function_name}],
            StartTime=start_time,
            EndTime=end_time,
            Period=3600,
            Statistics=['Sum']
        )
        
        # Throttle count
        throttle_response = self.cloudwatch.get_metric_statistics(
            Namespace='AWS/Lambda',
            MetricName='Throttles',
            Dimensions=[{'Name': 'FunctionName', 'Value': function_name}],
            StartTime=start_time,
            EndTime=end_time,
            Period=3600,
            Statistics=['Sum']
        )
        
        # Invocation count
        invocation_response = self.cloudwatch.get_metric_statistics(
            Namespace='AWS/Lambda',
            MetricName='Invocations',
            Dimensions=[{'Name': 'FunctionName', 'Value': function_name}],
            StartTime=start_time,
            EndTime=end_time,
            Period=3600,
            Statistics=['Sum']
        )
        
        # Process metrics
        avg_duration = statistics.mean([dp['Average'] for dp in duration_response['Datapoints']]) if duration_response['Datapoints'] else 0
        max_duration = max([dp['Maximum'] for dp in duration_response['Datapoints']]) if duration_response['Datapoints'] else 0
        
        avg_memory_used = statistics.mean([dp['Average'] for dp in memory_response['Datapoints']]) if memory_response['Datapoints'] else 0
        max_memory_used = max([dp['Maximum'] for dp in memory_response['Datapoints']]) if memory_response['Datapoints'] else 0
        
        error_count = sum([dp['Sum'] for dp in error_response['Datapoints']]) if error_response['Datapoints'] else 0
        throttle_count = sum([dp['Sum'] for dp in throttle_response['Datapoints']]) if throttle_response['Datapoints'] else 0
        invocation_count = sum([dp['Sum'] for dp in invocation_response['Datapoints']]) if invocation_response['Datapoints'] else 0
        
        # Calculate cost estimate
        gb_seconds = (memory_size / 1024) * (avg_duration / 1000) * invocation_count
        cost_estimate = (gb_seconds * self.price_per_gb_second) + (invocation_count * self.price_per_request)
        
        return LambdaMetrics(
            function_name=function_name,
            avg_duration=avg_duration,
            max_duration=max_duration,
            avg_memory_used=avg_memory_used,
            max_memory_used=max_memory_used,
            memory_size=memory_size,
            error_count=error_count,
            throttle_count=throttle_count,
            invocation_count=invocation_count,
            cost_estimate=cost_estimate
        )
    
    def analyze_performance(self, metrics: LambdaMetrics) -> List[PerformanceRecommendation]:
        """
        Analyze performance metrics and generate recommendations.
        
        Args:
            metrics: Lambda function metrics
            
        Returns:
            List of performance recommendations
        """
        recommendations = []
        
        # Memory analysis
        if metrics.avg_memory_used > 0:
            memory_utilization = (metrics.avg_memory_used / metrics.memory_size) * 100
            
            if memory_utilization > 90:
                recommendations.append(PerformanceRecommendation(
                    metric_type="memory",
                    current_value=metrics.memory_size,
                    recommended_value=min(metrics.memory_size * 1.5, 3008),
                    reasoning=f"Memory utilization is {memory_utilization:.1f}%, consider increasing memory",
                ))
            elif memory_utilization < 50:
                new_memory = max(int(metrics.memory_size * 0.75), 512)
                potential_savings = self._calculate_memory_savings(metrics, new_memory)
                recommendations.append(PerformanceRecommendation(
                    metric_type="memory",
                    current_value=metrics.memory_size,
                    recommended_value=new_memory,
                    reasoning=f"Memory utilization is {memory_utilization:.1f}%, consider reducing memory",
                    potential_savings=potential_savings
                ))
        
        # Duration analysis for query functions
        if "query" in metrics.function_name.lower():
            if metrics.avg_duration > 20000:  # 20 seconds
                recommendations.append(PerformanceRecommendation(
                    metric_type="timeout",
                    current_value=metrics.avg_duration,
                    recommended_value=30000,
                    reasoning="Average duration is close to timeout, consider optimizing or increasing timeout"
                ))
            elif metrics.max_duration > 25000:  # 25 seconds
                recommendations.append(PerformanceRecommendation(
                    metric_type="timeout",
                    current_value=metrics.max_duration,
                    recommended_value=35000,
                    reasoning="Maximum duration exceeds safe threshold, consider increasing timeout"
                ))
        
        # Error rate analysis
        if metrics.invocation_count > 0:
            error_rate = (metrics.error_count / metrics.invocation_count) * 100
            if error_rate > 5:
                recommendations.append(PerformanceRecommendation(
                    metric_type="error_rate",
                    current_value=error_rate,
                    recommended_value=2.0,
                    reasoning=f"Error rate is {error_rate:.1f}%, investigate and fix errors"
                ))
        
        # Throttling analysis
        if metrics.throttle_count > 0:
            recommendations.append(PerformanceRecommendation(
                metric_type="concurrency",
                current_value=metrics.throttle_count,
                recommended_value=0,
                reasoning=f"Function was throttled {metrics.throttle_count} times, consider increasing concurrency limit"
            ))
        
        return recommendations
    
    def _calculate_memory_savings(self, metrics: LambdaMetrics, new_memory: int) -> float:
        """Calculate potential cost savings from memory reduction."""
        current_gb_seconds = (metrics.memory_size / 1024) * (metrics.avg_duration / 1000) * metrics.invocation_count
        new_gb_seconds = (new_memory / 1024) * (metrics.avg_duration / 1000) * metrics.invocation_count
        
        current_cost = current_gb_seconds * self.price_per_gb_second
        new_cost = new_gb_seconds * self.price_per_gb_second
        
        return current_cost - new_cost
    
    def generate_report(self, function_names: List[str], hours: int = 24) -> Dict:
        """
        Generate a comprehensive performance report.
        
        Args:
            function_names: List of Lambda function names to analyze
            hours: Hours of metrics to analyze
            
        Returns:
            Performance report dictionary
        """
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "period_hours": hours,
            "functions": {},
            "summary": {
                "total_invocations": 0,
                "total_errors": 0,
                "total_throttles": 0,
                "total_cost_estimate": 0.0,
                "functions_needing_attention": []
            }
        }
        
        for function_name in function_names:
            try:
                metrics = self.get_function_metrics(function_name, hours)
                recommendations = self.analyze_performance(metrics)
                
                report["functions"][function_name] = {
                    "metrics": {
                        "avg_duration_ms": metrics.avg_duration,
                        "max_duration_ms": metrics.max_duration,
                        "avg_memory_used_mb": metrics.avg_memory_used,
                        "max_memory_used_mb": metrics.max_memory_used,
                        "memory_size_mb": metrics.memory_size,
                        "error_count": metrics.error_count,
                        "throttle_count": metrics.throttle_count,
                        "invocation_count": metrics.invocation_count,
                        "cost_estimate_usd": metrics.cost_estimate
                    },
                    "recommendations": [
                        {
                            "type": rec.metric_type,
                            "current": rec.current_value,
                            "recommended": rec.recommended_value,
                            "reasoning": rec.reasoning,
                            "savings": rec.potential_savings
                        }
                        for rec in recommendations
                    ]
                }
                
                # Update summary
                report["summary"]["total_invocations"] += metrics.invocation_count
                report["summary"]["total_errors"] += metrics.error_count
                report["summary"]["total_throttles"] += metrics.throttle_count
                report["summary"]["total_cost_estimate"] += metrics.cost_estimate
                
                if recommendations:
                    report["summary"]["functions_needing_attention"].append(function_name)
                    
            except Exception as e:
                report["functions"][function_name] = {
                    "error": f"Failed to get metrics: {str(e)}"
                }
        
        return report


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor Lambda function performance")
    parser.add_argument("--functions", nargs="+", 
                       default=["nice-cks-graphrag-dev-api-query", 
                               "nice-cks-graphrag-dev-api-health",
                               "nice-cks-graphrag-dev-sync"],
                       help="Lambda function names to monitor")
    parser.add_argument("--hours", type=int, default=24, 
                       help="Hours of metrics to analyze")
    parser.add_argument("--output", type=str, 
                       help="Output file for JSON report")
    
    args = parser.parse_args()
    
    monitor = LambdaPerformanceMonitor()
    report = monitor.generate_report(args.functions, args.hours)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {args.output}")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()