"""
Test runner script for the loan approval prediction system.
"""

import subprocess
import sys
from pathlib import Path
from loguru import logger


def run_tests():
    """Run all tests and generate coverage report."""
    logger.info("Starting test execution...")
    
    try:
        # Run tests with coverage
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/", 
            "-v", 
            "--cov=src", 
            "--cov-report=html",
            "--cov-report=term-missing"
        ], capture_output=True, text=True, cwd=Path.cwd())
        
        print("TEST EXECUTION RESULTS")
        print("=" * 50)
        print(result.stdout)
        
        if result.stderr:
            print("ERRORS/WARNINGS:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ All tests passed successfully!")
            print("📊 Coverage report generated in htmlcov/index.html")
        else:
            print("❌ Some tests failed!")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        print(f"❌ Test execution failed: {e}")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
