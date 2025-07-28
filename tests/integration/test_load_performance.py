#!/usr/bin/env python3
"""
Load testing and performance benchmarking for Care-GraphRAG system.
TASK-031: Test system performance under load conditions.

Tests concurrent query handling, sustained load performance, 
memory usage patterns, and throughput benchmarks.
"""

import unittest
import pytest
import time
import statistics
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple
import psutil
import os
from datetime import datetime, timedelta

from tests.fixtures.integration_data import PERFORMANCE_TEST_DATA
from src.qa_chain import QAChain
from src.hybrid_retriever import HybridRetriever
from src.monitoring.cost_tracker import CostTracker
from config.settings import get_settings
from config.logging import setup_logging, get_logger

# Setup logging for performance tests
setup_logging()
logger = get_logger(__name__)


class PerformanceMetrics:
    """Helper class to collect and analyze performance metrics."""
    
    def __init__(self):
        self.response_times = []
        self.errors = []
        self.start_time = None
        self.end_time = None
        self.memory_samples = []
        self.cpu_samples = []
        
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.response_times = []
        self.errors = []
        self.memory_samples = []
        self.cpu_samples = []
        
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.end_time = time.time()
        
    def record_response_time(self, duration: float):
        """Record a response time."""
        self.response_times.append(duration)
        
    def record_error(self, error: str):
        """Record an error."""
        self.errors.append(error)
        
    def sample_system_resources(self):
        """Sample current system resource usage."""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()
            
            self.memory_samples.append(memory_mb)
            self.cpu_samples.append(cpu_percent)
        except Exception as e:
            logger.warning(f"Failed to sample system resources: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        if not self.response_times:
            return {"error": "No response times recorded"}
            
        total_duration = self.end_time - self.start_time if self.end_time else 0
        
        stats = {
            "total_requests": len(self.response_times),
            "total_errors": len(self.errors),
            "error_rate": len(self.errors) / (len(self.response_times) + len(self.errors)) if (len(self.response_times) + len(self.errors)) > 0 else 0,
            "total_duration": total_duration,
            "requests_per_second": len(self.response_times) / total_duration if total_duration > 0 else 0,
            "response_times": {
                "min": min(self.response_times),
                "max": max(self.response_times),
                "mean": statistics.mean(self.response_times),
                "median": statistics.median(self.response_times),
                "p95": statistics.quantiles(self.response_times, n=20)[18] if len(self.response_times) >= 20 else max(self.response_times),
                "p99": statistics.quantiles(self.response_times, n=100)[98] if len(self.response_times) >= 100 else max(self.response_times),
            }
        }
        
        if self.memory_samples:
            stats["memory_mb"] = {
                "min": min(self.memory_samples),
                "max": max(self.memory_samples),
                "mean": statistics.mean(self.memory_samples)
            }
            
        if self.cpu_samples:
            stats["cpu_percent"] = {
                "min": min(self.cpu_samples),
                "max": max(self.cpu_samples),
                "mean": statistics.mean(self.cpu_samples)
            }
            
        return stats


class TestConcurrentLoad(unittest.TestCase):
    """Test concurrent request handling and load capacity."""

    @classmethod
    def setUpClass(cls):
        """Set up concurrent load test environment."""
        logger.info("Setting up concurrent load test environment")
        
        try:
            # Initialize system components
            cls.settings = get_settings()
            cls.retriever = HybridRetriever(
                max_depth=2,  # Reduced for performance
                similarity_threshold=0.6,
                max_results=5,
                monitoring_enabled=True
            )
            cls.qa_chain = QAChain(
                retriever=cls.retriever,
                cost_tracking=True
            )
            
            # Test queries for concurrent execution
            cls.test_queries = PERFORMANCE_TEST_DATA["concurrent_queries"][:10]  # Limit for testing
            
            logger.info("Concurrent load test environment ready")
            
        except Exception as e:
            logger.error(f"Failed to setup concurrent load test environment: {e}")
            raise

    def test_concurrent_query_handling(self):
        """Test handling of concurrent queries."""
        logger.info("Testing concurrent query handling")
        
        metrics = PerformanceMetrics()
        num_concurrent = 5  # Moderate concurrency for testing
        
        def execute_query(query_id: int, query: str) -> Tuple[int, float, str]:
            """Execute a single query and return results."""
            start_time = time.time()
            try:
                result = self.qa_chain.ask(
                    question=query,
                    return_source_documents=True,
                    max_sources=3
                )
                duration = time.time() - start_time
                return query_id, duration, "success"
            except Exception as e:
                duration = time.time() - start_time
                return query_id, duration, f"error: {str(e)}"
        
        # Start monitoring
        metrics.start_monitoring()
        
        # Execute concurrent queries
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = []
            
            for i in range(num_concurrent):
                query = self.test_queries[i % len(self.test_queries)]
                future = executor.submit(execute_query, i, query)
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures, timeout=60):  # 60 second timeout
                try:
                    query_id, duration, status = future.result()
                    
                    if status == "success":
                        metrics.record_response_time(duration)
                    else:
                        metrics.record_error(status)
                        
                    # Sample system resources periodically
                    if query_id % 2 == 0:
                        metrics.sample_system_resources()
                        
                except Exception as e:
                    metrics.record_error(f"Future exception: {e}")
        
        # Stop monitoring
        metrics.stop_monitoring()
        
        # Analyze results
        stats = metrics.get_statistics()
        
        # Validate concurrent performance
        self.assertGreater(stats["total_requests"], 0, "Should have processed some requests")
        self.assertLessEqual(stats["error_rate"], 0.2, "Error rate should be reasonable (<20%)")
        
        # Log performance metrics
        logger.info(f"Concurrent Query Performance:")
        logger.info(f"  Total requests: {stats['total_requests']}")
        logger.info(f"  Error rate: {stats['error_rate']:.2%}")
        logger.info(f"  Requests/second: {stats['requests_per_second']:.2f}")
        logger.info(f"  Response times: {stats['response_times']['mean']:.2f}s avg, {stats['response_times']['p95']:.2f}s p95")
        
        if "memory_mb" in stats:
            logger.info(f"  Memory usage: {stats['memory_mb']['mean']:.1f}MB avg")

    def test_sustained_load_performance(self):
        """Test performance under sustained load over time."""
        logger.info("Testing sustained load performance")
        
        metrics = PerformanceMetrics()
        duration_seconds = 30  # 30 second sustained test
        queries_per_second = 2  # Moderate sustained rate
        
        def sustained_query_worker(stop_event: threading.Event):
            """Worker thread for sustained query execution."""
            query_count = 0
            while not stop_event.is_set():
                try:
                    query = self.test_queries[query_count % len(self.test_queries)]
                    
                    start_time = time.time()
                    result = self.qa_chain.ask(
                        question=query,
                        return_source_documents=True,
                        max_sources=3
                    )
                    duration = time.time() - start_time
                    
                    metrics.record_response_time(duration)
                    query_count += 1
                    
                    # Control rate
                    time.sleep(1.0 / queries_per_second)
                    
                except Exception as e:
                    metrics.record_error(f"Query error: {e}")
                    time.sleep(0.5)  # Brief pause on error
        
        # Start monitoring
        metrics.start_monitoring()
        
        # Create stop event
        stop_event = threading.Event()
        
        # Start worker threads
        num_workers = 2
        workers = []
        
        for i in range(num_workers):
            worker = threading.Thread(target=sustained_query_worker, args=(stop_event,))
            worker.daemon = True
            worker.start()
            workers.append(worker)
        
        # Run sustained load test
        resource_sampling_thread = threading.Thread(
            target=self._sample_resources_continuously,
            args=(metrics, stop_event, 2.0)  # Sample every 2 seconds
        )
        resource_sampling_thread.daemon = True
        resource_sampling_thread.start()
        
        # Let test run for specified duration
        time.sleep(duration_seconds)
        
        # Stop workers
        stop_event.set()
        
        # Wait for workers to finish
        for worker in workers:
            worker.join(timeout=5)
        
        resource_sampling_thread.join(timeout=2)
        
        # Stop monitoring
        metrics.stop_monitoring()
        
        # Analyze sustained performance
        stats = metrics.get_statistics()
        
        # Validate sustained performance
        expected_min_requests = duration_seconds * queries_per_second * num_workers * 0.7  # 70% efficiency
        self.assertGreater(stats["total_requests"], expected_min_requests,
                          f"Should process at least {expected_min_requests} requests")
        
        self.assertLessEqual(stats["error_rate"], 0.1, "Sustained error rate should be low (<10%)")
        
        # Log sustained performance metrics
        logger.info(f"Sustained Load Performance ({duration_seconds}s):")
        logger.info(f"  Total requests: {stats['total_requests']}")
        logger.info(f"  Requests/second: {stats['requests_per_second']:.2f}")
        logger.info(f"  Error rate: {stats['error_rate']:.2%}")
        logger.info(f"  Response times: {stats['response_times']['mean']:.2f}s avg, {stats['response_times']['max']:.2f}s max")
        
        if "memory_mb" in stats:
            logger.info(f"  Memory: {stats['memory_mb']['min']:.1f}-{stats['memory_mb']['max']:.1f}MB")
        if "cpu_percent" in stats:
            logger.info(f"  CPU: {stats['cpu_percent']['mean']:.1f}% avg")

    def _sample_resources_continuously(self, metrics: PerformanceMetrics, 
                                     stop_event: threading.Event, 
                                     interval: float):
        """Continuously sample system resources."""
        while not stop_event.is_set():
            metrics.sample_system_resources()
            time.sleep(interval)

    def test_memory_usage_under_load(self):
        """Test memory usage patterns under load."""
        logger.info("Testing memory usage under load")
        
        # Get baseline memory usage
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        logger.info(f"Baseline memory usage: {baseline_memory:.1f}MB")
        
        metrics = PerformanceMetrics()
        metrics.start_monitoring()
        
        # Execute queries while monitoring memory
        num_queries = 20
        memory_samples = []
        
        for i in range(num_queries):
            try:
                query = self.test_queries[i % len(self.test_queries)]
                
                # Sample memory before query
                pre_memory = process.memory_info().rss / 1024 / 1024
                
                # Execute query
                start_time = time.time()
                result = self.qa_chain.ask(
                    question=query,
                    return_source_documents=True
                )
                duration = time.time() - start_time
                
                # Sample memory after query
                post_memory = process.memory_info().rss / 1024 / 1024
                
                memory_samples.append({
                    "query_id": i,
                    "pre_memory": pre_memory,
                    "post_memory": post_memory,
                    "memory_delta": post_memory - pre_memory,
                    "duration": duration
                })
                
                metrics.record_response_time(duration)
                
                # Brief pause between queries
                time.sleep(0.1)
                
            except Exception as e:
                metrics.record_error(f"Memory test query error: {e}")
        
        metrics.stop_monitoring()
        
        # Analyze memory usage
        if memory_samples:
            final_memory = memory_samples[-1]["post_memory"]
            total_memory_increase = final_memory - baseline_memory
            avg_memory_delta = statistics.mean([s["memory_delta"] for s in memory_samples])
            max_memory_delta = max([s["memory_delta"] for s in memory_samples])
            
            # Validate memory usage
            self.assertLess(total_memory_increase, 500,  # 500MB limit
                           f"Total memory increase should be reasonable: {total_memory_increase:.1f}MB")
            
            self.assertLess(max_memory_delta, 100,  # 100MB per query limit
                           f"Per-query memory spike should be reasonable: {max_memory_delta:.1f}MB")
            
            logger.info(f"Memory Usage Analysis:")
            logger.info(f"  Baseline: {baseline_memory:.1f}MB")
            logger.info(f"  Final: {final_memory:.1f}MB")
            logger.info(f"  Total increase: {total_memory_increase:.1f}MB")
            logger.info(f"  Avg per-query delta: {avg_memory_delta:.2f}MB")
            logger.info(f"  Max per-query delta: {max_memory_delta:.1f}MB")


class TestThroughputBenchmarks(unittest.TestCase):
    """Test system throughput and capacity benchmarks."""

    @classmethod
    def setUpClass(cls):
        """Set up throughput benchmark environment."""
        logger.info("Setting up throughput benchmark environment")
        
        cls.retriever = HybridRetriever(
            max_depth=2,
            similarity_threshold=0.6,
            max_results=5
        )
        cls.qa_chain = QAChain(retriever=cls.retriever)
        
        cls.benchmark_queries = [
            "What is first-line treatment for hypertension?",
            "Blood pressure targets for diabetes?",
            "ACE inhibitor side effects?",
            "When to use combination therapy?",
            "Lifestyle advice for blood pressure?"
        ]

    def test_single_threaded_throughput(self):
        """Test single-threaded query throughput."""
        logger.info("Testing single-threaded throughput")
        
        metrics = PerformanceMetrics()
        metrics.start_monitoring()
        
        num_queries = 10  # Reduced for testing
        
        for i in range(num_queries):
            query = self.benchmark_queries[i % len(self.benchmark_queries)]
            
            start_time = time.time()
            try:
                result = self.qa_chain.ask(
                    question=query,
                    return_source_documents=True,
                    max_sources=3
                )
                duration = time.time() - start_time
                metrics.record_response_time(duration)
                
            except Exception as e:
                duration = time.time() - start_time
                metrics.record_error(f"Query error: {e}")
            
            # Sample resources periodically
            if i % 3 == 0:
                metrics.sample_system_resources()
        
        metrics.stop_monitoring()
        
        # Analyze throughput
        stats = metrics.get_statistics()
        
        # Validate single-threaded performance
        self.assertGreater(stats["requests_per_second"], 0.5,
                          "Should process at least 0.5 queries per second")
        
        self.assertLess(stats["response_times"]["mean"], 4.0,
                       "Average response time should be under 4 seconds")
        
        logger.info(f"Single-threaded Throughput:")
        logger.info(f"  Queries/second: {stats['requests_per_second']:.2f}")
        logger.info(f"  Avg response time: {stats['response_times']['mean']:.2f}s")
        logger.info(f"  P95 response time: {stats['response_times']['p95']:.2f}s")

    def test_multi_threaded_throughput(self):
        """Test multi-threaded query throughput."""
        logger.info("Testing multi-threaded throughput")
        
        metrics = PerformanceMetrics()
        num_threads = 3
        queries_per_thread = 5
        
        def thread_worker(thread_id: int):
            """Worker function for throughput testing."""
            for i in range(queries_per_thread):
                query = self.benchmark_queries[(thread_id * queries_per_thread + i) % len(self.benchmark_queries)]
                
                start_time = time.time()
                try:
                    result = self.qa_chain.ask(
                        question=query,
                        return_source_documents=True,
                        max_sources=3
                    )
                    duration = time.time() - start_time
                    metrics.record_response_time(duration)
                    
                except Exception as e:
                    duration = time.time() - start_time
                    metrics.record_error(f"Thread {thread_id} error: {e}")
        
        # Start monitoring
        metrics.start_monitoring()
        
        # Create and start threads
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=thread_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=120)  # 2 minute timeout per thread
        
        metrics.stop_monitoring()
        
        # Analyze multi-threaded throughput
        stats = metrics.get_statistics()
        
        expected_queries = num_threads * queries_per_thread
        
        # Validate multi-threaded performance
        completion_rate = stats["total_requests"] / expected_queries
        self.assertGreater(completion_rate, 0.8,
                          f"Should complete most queries: {completion_rate:.2%}")
        
        # Should achieve better throughput than single-threaded
        self.assertGreater(stats["requests_per_second"], 0.8,
                          "Multi-threaded throughput should be reasonable")
        
        logger.info(f"Multi-threaded Throughput ({num_threads} threads):")
        logger.info(f"  Queries/second: {stats['requests_per_second']:.2f}")
        logger.info(f"  Completion rate: {completion_rate:.2%}")
        logger.info(f"  Error rate: {stats['error_rate']:.2%}")
        logger.info(f"  Avg response time: {stats['response_times']['mean']:.2f}s")

    def test_burst_load_handling(self):
        """Test handling of burst load scenarios."""
        logger.info("Testing burst load handling")
        
        metrics = PerformanceMetrics()
        
        # Simulate burst: rapid queries followed by quiet period
        burst_size = 8
        burst_queries = self.benchmark_queries * 2  # More variety
        
        metrics.start_monitoring()
        
        # Burst phase
        logger.info(f"Starting burst of {burst_size} queries")
        
        with ThreadPoolExecutor(max_workers=burst_size) as executor:
            futures = []
            
            for i in range(burst_size):
                query = burst_queries[i % len(burst_queries)]
                future = executor.submit(self._execute_timed_query, query, i)
                futures.append(future)
            
            # Collect burst results
            for future in as_completed(futures, timeout=60):
                try:
                    duration, status = future.result()
                    if status == "success":
                        metrics.record_response_time(duration)
                    else:
                        metrics.record_error(status)
                except Exception as e:
                    metrics.record_error(f"Burst future error: {e}")
        
        logger.info("Burst phase completed, starting quiet phase")
        
        # Quiet phase - single queries with pauses
        for i in range(3):
            query = self.benchmark_queries[i]
            duration, status = self._execute_timed_query(query, f"quiet_{i}")
            
            if status == "success":
                metrics.record_response_time(duration)
            else:
                metrics.record_error(status)
            
            time.sleep(1.0)  # Pause between quiet queries
        
        metrics.stop_monitoring()
        
        # Analyze burst handling
        stats = metrics.get_statistics()
        
        # Validate burst handling
        self.assertGreater(stats["total_requests"], burst_size,
                          "Should handle most burst queries")
        
        self.assertLessEqual(stats["error_rate"], 0.3,
                           "Burst error rate should be acceptable (<30%)")
        
        logger.info(f"Burst Load Handling:")
        logger.info(f"  Total requests: {stats['total_requests']}")
        logger.info(f"  Burst + quiet success rate: {1 - stats['error_rate']:.2%}")
        logger.info(f"  Peak throughput: {stats['requests_per_second']:.2f} queries/sec")
        logger.info(f"  Response time range: {stats['response_times']['min']:.2f}-{stats['response_times']['max']:.2f}s")

    def _execute_timed_query(self, query: str, query_id: Any) -> Tuple[float, str]:
        """Execute a query and return timing information."""
        start_time = time.time()
        try:
            result = self.qa_chain.ask(
                question=query,
                return_source_documents=True,
                max_sources=3
            )
            duration = time.time() - start_time
            return duration, "success"
        except Exception as e:
            duration = time.time() - start_time
            return duration, f"error: {str(e)}"


class TestPerformanceBenchmarks(unittest.TestCase):
    """Test against defined performance benchmarks from test fixtures."""

    @classmethod
    def setUpClass(cls):
        """Set up performance benchmark tests."""
        logger.info("Setting up performance benchmark tests")
        
        cls.expected_performance = PERFORMANCE_TEST_DATA["expected_performance"]
        cls.retriever = HybridRetriever()
        cls.qa_chain = QAChain(retriever=cls.retriever)

    def test_benchmark_compliance(self):
        """Test compliance with defined performance benchmarks."""
        logger.info("Testing compliance with performance benchmarks")
        
        metrics = PerformanceMetrics()
        test_queries = [
            "What is first-line treatment for hypertension?",
            "Blood pressure monitoring frequency?",
            "ACE inhibitor contraindications?"
        ]
        
        metrics.start_monitoring()
        
        # Execute benchmark queries
        for query in test_queries:
            try:
                start_time = time.time()
                result = self.qa_chain.ask(question=query, return_source_documents=True)
                duration = time.time() - start_time
                
                metrics.record_response_time(duration)
                
            except Exception as e:
                metrics.record_error(f"Benchmark query error: {e}")
        
        metrics.stop_monitoring()
        
        # Compare against benchmarks
        stats = metrics.get_statistics()
        expected = self.expected_performance
        
        # Relaxed benchmarks for testing environment
        avg_response_time = stats["response_times"]["mean"]
        p95_response_time = stats["response_times"]["p95"]
        
        self.assertLess(avg_response_time, expected["average_response_time"] * 2,
                       f"Average response time within 2x benchmark: {avg_response_time:.2f}s")
        
        self.assertLess(p95_response_time, expected["95th_percentile_response_time"] * 2,
                       f"P95 response time within 2x benchmark: {p95_response_time:.2f}s")
        
        self.assertLessEqual(stats["error_rate"], expected["error_rate"] * 2,
                           f"Error rate within 2x benchmark: {stats['error_rate']:.2%}")
        
        logger.info(f"Performance Benchmark Results:")
        logger.info(f"  Avg response time: {avg_response_time:.2f}s (target: {expected['average_response_time']:.1f}s)")
        logger.info(f"  P95 response time: {p95_response_time:.2f}s (target: {expected['95th_percentile_response_time']:.1f}s)")
        logger.info(f"  Error rate: {stats['error_rate']:.2%} (target: {expected['error_rate']:.1%})")


if __name__ == '__main__':
    # Configure test runner for performance tests
    pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '--durations=0',  # Show all test durations
        '--junit-xml=test_results/integration_performance.xml'
    ])