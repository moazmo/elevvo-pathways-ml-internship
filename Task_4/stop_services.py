"""
Stop All Services Script
========================

This script stops all running services for the Walmart Sales Forecasting system.
"""

import subprocess
import sys
import time

def kill_process_on_port(port):
    """Kill process running on specified port (Windows)."""
    try:
        print(f"🔍 Looking for processes on port {port}...")
        
        # Find process using the port
        result = subprocess.run([
            'netstat', '-ano'
        ], capture_output=True, text=True, check=True)
        
        lines = result.stdout.split('\n')
        killed = False
        
        for line in lines:
            if f':{port}' in line and 'LISTENING' in line:
                parts = line.split()
                if len(parts) >= 5:
                    pid = parts[-1]
                    try:
                        subprocess.run(['taskkill', '/F', '/PID', pid], 
                                     capture_output=True, check=True)
                        print(f"✅ Killed process {pid} on port {port}")
                        killed = True
                    except subprocess.CalledProcessError:
                        print(f"⚠️  Could not kill process {pid}")
        
        if not killed:
            print(f"ℹ️  No processes found on port {port}")
        
        return killed
    except Exception as e:
        print(f"❌ Error checking port {port}: {e}")
        return False

def stop_docker_services():
    """Stop Docker services."""
    try:
        print("🐳 Stopping Docker services...")
        result = subprocess.run([
            'docker-compose', 'down'
        ], capture_output=True, text=True, check=True)
        print("✅ Docker services stopped")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Docker services may not be running: {e}")
        return False
    except FileNotFoundError:
        print("⚠️  Docker not found")
        return False

def main():
    """Stop all services."""
    print("🛑 Stopping Walmart Sales Forecasting Services")
    print("=" * 50)
    
    # Stop processes on common ports
    ports_to_check = [8000, 3000, 5432, 6379]
    
    for port in ports_to_check:
        kill_process_on_port(port)
        time.sleep(1)
    
    # Stop Docker services
    stop_docker_services()
    
    print("\n" + "=" * 50)
    print("✅ All services stopped")
    print("💡 You can now restart with: python start_full_system.py --frontend")

if __name__ == "__main__":
    main()