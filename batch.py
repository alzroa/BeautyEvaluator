#!/usr/bin/env python3
"""
BeautyEvaluator Batch Processor
Run with: python batch.py ./photos/ --output results.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from main import BeautyAnalyzer


def analyze_single(image_path, analyzer):
    """Analyze a single image."""
    try:
        result = analyzer.analyze(image_path, draw_overlay=False, detailed=False)
        if result:
            return {
                'image': str(image_path),
                'beauty_score': result['beauty_score'],
                'symmetry': result['symmetry'],
                'golden_ratio': result['golden_ratio'],
                'success': True
            }
        return {
            'image': str(image_path),
            'success': False,
            'error': 'No face detected'
        }
    except Exception as e:
        return {
            'image': str(image_path),
            'success': False,
            'error': str(e)
        }


def find_images(directory, extensions=None):
    """Find all image files in directory."""
    if extensions is None:
        extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    
    directory = Path(directory)
    images = []
    
    for ext in extensions:
        images.extend(directory.glob(f'*{ext}'))
        images.extend(directory.glob(f'*{ext.upper()}'))
    
    return sorted(images)


def main():
    parser = argparse.ArgumentParser(description='BeautyEvaluator Batch Processor')
    parser.add_argument('directory', help='Directory containing images')
    parser.add_argument('--output', '-o', default='batch_results.json',
                       help='Output JSON file (default: batch_results.json)')
    parser.add_argument('--threads', '-t', type=int, default=4,
                       help='Number of parallel threads (default: 4)')
    parser.add_argument('--extensions', nargs='+',
                       default=['jpg', 'jpeg', 'png', 'webp'],
                       help='File extensions to process')
    
    args = parser.parse_args()
    
    # Find images
    images = find_images(args.directory, set(f'.{e}' for e in args.extensions))
    
    if not images:
        print(f"❌ No images found in {args.directory}")
        return 1
    
    print(f"📁 Found {len(images)} images")
    print(f"🔄 Processing with {args.threads} threads...")
    
    # Initialize analyzer
    analyzer = BeautyAnalyzer()
    
    # Process images in parallel
    results = []
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {executor.submit(analyze_single, img, analyzer): img for img in images}
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            results.append(result)
            
            status = "✅" if result.get('success') else "❌"
            img_name = Path(result['image']).name
            score = result.get('beauty_score', 'N/A')
            print(f"  {status} [{completed}/{len(images)}] {img_name}: {score}")
    
    # Sort by beauty score
    results.sort(key=lambda x: x.get('beauty_score', 0), reverse=True)
    
    # Summary
    successful = [r for r in results if r.get('success')]
    failed = [r for r in results if not r.get('success')]
    
    print(f"\n📊 Batch Processing Complete!")
    print(f"  ✅ Successful: {len(successful)}")
    print(f"  ❌ Failed: {len(failed)}")
    
    if successful:
        scores = [r['beauty_score'] for r in successful]
        avg = sum(scores) / len(scores)
        print(f"  📈 Average Score: {avg:.1f}")
        print(f"  🏆 Highest: {max(scores)}")
        print(f"  📉 Lowest: {min(scores)}")
    
    # Save results
    output = {
        'summary': {
            'total': len(images),
            'successful': len(successful),
            'failed': len(failed),
            'average_score': round(sum(r['beauty_score'] for r in successful) / len(successful), 1) if successful else 0,
            'highest_score': max((r['beauty_score'] for r in successful), default=0),
            'lowest_score': min((r['beauty_score'] for r in successful), default=0)
        },
        'results': results
    }
    
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Results saved to: {args.output}")
    return 0


if __name__ == '__main__':
    sys.exit(main())