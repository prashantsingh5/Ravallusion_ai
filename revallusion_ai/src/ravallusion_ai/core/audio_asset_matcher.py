"""
Audio Asset Matching Module
Detects which audio assets from the assets folder are used in the student video
Using MFCC correlation and audio fingerprinting techniques
"""

import os
import sys
import subprocess
import librosa
import numpy as np
import shutil
import glob
from scipy import signal
from scipy.signal import find_peaks
from collections import defaultdict
import hashlib
import warnings
warnings.filterwarnings('ignore')

def extract_background_audio(video_path, output_name=None):
    """
    Extract background audio from video using AI vocal separation (Demucs)
    Same function as in final_working.ipynb
    """
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return None
    
    if output_name is None:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_name = f"{base_name}_background.wav"
    
    temp_audio = "temp_extracted_audio.wav"
    
    try:
        print(f"Extracting audio from: {video_path}")
        # Extract audio using FFmpeg
        cmd = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", 
               "-ar", "44100", "-ac", "2", "-y", temp_audio]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            return None
        
        print("🤖 Separating vocals using AI (Demucs)...")
        # Use Demucs to separate vocals from background
        cmd = ["python", "-m", "demucs.separate", "--two-stems", "vocals", temp_audio]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        
        if result.returncode != 0:
            print(f"Demucs error: {result.stderr}")
            return None
        
        # Move the background audio to final location
        separated_path = os.path.join("separated", "htdemucs", 
                                    os.path.splitext(temp_audio)[0], "no_vocals.wav")
        
        if os.path.exists(separated_path):
            shutil.move(separated_path, output_name)
            print(f"Background audio saved as: {output_name}")
            
            # Cleanup
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            if os.path.exists("separated"):
                shutil.rmtree("separated")
            
            return output_name
        else:
            print(f"Separated audio not found at: {separated_path}")
            return None
            
    except Exception as e:
        print(f"Error during extraction: {str(e)}")
        return None

def is_audio_present(clip_path, main_audio_path, verbose=False):
    """
    Enhanced MFCC detection - EXACT SAME as final_working.ipynb
    Uses improved MFCC correlation with sensitive thresholds optimized for present audio
    """
    try:
        # Load audio files
        clip_audio, sr1 = librosa.load(clip_path, sr=22050)
        main_audio, sr2 = librosa.load(main_audio_path, sr=22050)
        
        if verbose:
            print(f"Clip duration: {len(clip_audio)/sr1:.2f}s")
            print(f"Main audio duration: {len(main_audio)/sr2:.2f}s")
        
        # Skip if clip is longer than main audio
        if len(clip_audio) >= len(main_audio):
            if verbose:
                print("Clip is longer than or equal to main audio")
            return False, "exceeds_video_length"
        
        # Extract MFCC features
        if verbose:
            print("Extracting MFCC features...")
        
        clip_mfcc = librosa.feature.mfcc(y=clip_audio, sr=sr1, n_mfcc=13, hop_length=512)
        main_mfcc = librosa.feature.mfcc(y=main_audio, sr=sr2, n_mfcc=13, hop_length=512)
        
        # FIXED: Enhanced Cross-correlation with MFCC relationship preservation
        def normalized_cross_correlation(clip_features, main_features):
            correlations = []
            window_size = clip_features.shape[1]
            
            # Process each position in the main audio
            for i in range(main_features.shape[1] - window_size + 1):
                window = main_features[:, i:i+window_size]
                
                # Calculate correlation coefficient for each MFCC coefficient
                mfcc_correlations = []
                for mfcc_idx in range(clip_features.shape[0]):
                    clip_mfcc_coef = clip_features[mfcc_idx, :]
                    window_mfcc_coef = window[mfcc_idx, :]
                    
                    # Normalize
                    clip_norm = (clip_mfcc_coef - np.mean(clip_mfcc_coef)) / (np.std(clip_mfcc_coef) + 1e-8)
                    window_norm = (window_mfcc_coef - np.mean(window_mfcc_coef)) / (np.std(window_mfcc_coef) + 1e-8)
                    
                    # Calculate correlation
                    corr = np.corrcoef(clip_norm, window_norm)[0, 1]
                    if not np.isnan(corr):
                        mfcc_correlations.append(corr)
                
                # Average correlation across all MFCC coefficients
                if len(mfcc_correlations) > 0:
                    avg_corr = np.mean(mfcc_correlations)
                    correlations.append(avg_corr)
                else:
                    correlations.append(0)
            
            return np.array(correlations)
        
        # Template matching with SENSITIVE validation for present audio
        def template_matching_validation(clip_features, main_features):
            correlations = normalized_cross_correlation(clip_features, main_features)
            
            if len(correlations) == 0:
                return False, 0, {}
            
            # Statistical analysis
            max_corr = np.max(correlations)
            mean_corr = np.mean(correlations)
            std_corr = np.std(correlations)
            
            # Z-score of the maximum correlation
            z_score = (max_corr - mean_corr) / (std_corr + 1e-8)
            
            # Multiple validation criteria
            validation = {
                'max_correlation': max_corr,
                'mean_correlation': mean_corr,
                'std_correlation': std_corr,
                'z_score': z_score,
                'num_high_peaks': np.sum(correlations > mean_corr + 2 * std_corr),
                'peak_ratio': max_corr / (mean_corr + 1e-8)
            }
            
            # SENSITIVE detection criteria optimized for present audio (OR logic)
            is_detected = (
                max_corr > 0.4 or               # Lower correlation threshold (was 0.7)
                z_score > 2.5 or               # Lower z-score threshold (was 4.0)  
                validation['peak_ratio'] > 2.0    # Lower peak ratio (was 3.0)
            )
            
            return is_detected, max_corr, validation
        
        # Perform detection
        if verbose:
            print("Performing detection...")
        
        detected, confidence, stats = template_matching_validation(clip_mfcc, main_mfcc)
        
        # Print results (EXACT same as notebook)
        if verbose:
            print("\\n" + "="*50)
            print("🎵 ENHANCED MFCC DETECTION RESULTS")
            print("="*50)
            print(f"Audio present: {'YES' if detected else '❌ NO'}")
            print(f"Confidence: {confidence:.4f}")
            print(f"Z-score: {stats['z_score']:.2f}")
            print(f"Peak ratio: {stats['peak_ratio']:.2f}")
            print(f"High peaks count: {stats['num_high_peaks']}")
            
            if not detected:
                reasons = []
                if confidence <= 0.4:
                    reasons.append(f"Low correlation ({confidence:.3f} <= 0.4)")
                if stats['z_score'] <= 2.5:
                    reasons.append(f"Weak peak significance ({stats['z_score']:.1f} <= 2.5)")
                if stats['peak_ratio'] <= 2.0:
                    reasons.append(f"Low peak ratio ({stats['peak_ratio']:.1f} <= 2.0)")
                
                print("Rejection reasons:")
                for reason in reasons:
                    print(f"  • {reason}")
        
        return detected, confidence
        
    except Exception as e:
        if verbose:
            print(f"MFCC Error: {e}")
        return False, 0

class AudioFingerprinter:
    """
    Audio Fingerprinting System - Same as final_working.ipynb
    Shazam-style detection using spectral peaks
    """
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self.window_size = 4096
        self.hop_length = self.window_size // 4
        
    def load_audio(self, file_path):
        """Load audio file"""
        try:
            audio, _ = librosa.load(file_path, sr=self.sample_rate)
            return audio
        except Exception as e:
            return None
    
    def get_spectrogram(self, audio):
        """Get magnitude spectrogram"""
        stft = librosa.stft(audio, n_fft=self.window_size, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        return magnitude
    
    def find_peaks_in_spectrum(self, magnitude_spectrum):
        """Find spectral peaks for fingerprinting - OPTIMIZED for efficiency"""
        peaks_list = []
        
        # Process every 4th frame for 4x speed improvement
        for time_idx in range(0, magnitude_spectrum.shape[1], 4):
            spectrum_slice = magnitude_spectrum[:, time_idx]
            
            # Find peaks in this time slice
            peaks, properties = find_peaks(spectrum_slice, 
                                         height=np.mean(spectrum_slice) + 2*np.std(spectrum_slice),
                                         distance=20)
            
            # Get the strongest peaks
            if len(peaks) > 0:
                peak_heights = spectrum_slice[peaks]
                # Sort by height and take top peaks
                top_indices = np.argsort(peak_heights)[-5:]  # Top 5 peaks
                top_peaks = peaks[top_indices]
                
                for peak_freq in top_peaks:
                    peaks_list.append((time_idx, peak_freq, spectrum_slice[peak_freq]))
        
        return peaks_list
    
    def create_fingerprint_hashes(self, peaks_list):
        """Create fingerprint hashes from peaks"""
        peaks_list = sorted(peaks_list, key=lambda x: x[0])  # Sort by time
        fingerprints = {}
        
        for i, (t1, f1, amp1) in enumerate(peaks_list):
            # Look for nearby peaks to create pairs
            for j in range(i+1, min(i+10, len(peaks_list))):
                t2, f2, amp2 = peaks_list[j]
                
                # Time difference constraint
                dt = t2 - t1
                if dt < 1 or dt > 10:
                    continue
                
                # Create hash from frequency pair and time difference
                hash_input = f"{int(f1)}_{int(f2)}_{int(dt)}"
                hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:16]
                
                # Store with time offset
                if hash_value not in fingerprints:
                    fingerprints[hash_value] = []
                fingerprints[hash_value].append(t1)
        
        return fingerprints
    
    def generate_fingerprint(self, audio):
        """Generate complete fingerprint for audio"""
        if audio is None or len(audio) == 0:
            return {}
        
        magnitude = self.get_spectrogram(audio)
        peaks = self.find_peaks_in_spectrum(magnitude)
        fingerprints = self.create_fingerprint_hashes(peaks)
        
        return fingerprints
    
    def match_fingerprints(self, clip_fingerprints, main_fingerprints, min_matches=2):
        """Match fingerprints and find time offsets - SENSITIVE for present audio"""
        matches = defaultdict(int)
        match_details = defaultdict(list)
        
        for hash_val, clip_times in clip_fingerprints.items():
            if hash_val in main_fingerprints:
                main_times = main_fingerprints[hash_val]
                
                for clip_time in clip_times:
                    for main_time in main_times:
                        offset = main_time - clip_time
                        matches[offset] += 1
                        match_details[offset].append({
                            'hash': hash_val,
                            'clip_time': clip_time,
                            'main_time': main_time
                        })
        
        if not matches:
            return False, 0, None, {}
        
        # Find the best offset (most matches)
        best_offset = max(matches.keys(), key=lambda x: matches[x])
        best_match_count = matches[best_offset]
        
        # Calculate confidence based on matches
        total_clip_hashes = len(clip_fingerprints)
        confidence = best_match_count / max(total_clip_hashes, 1)
        
        # SENSITIVE detection criteria for present audio
        is_match = (
            best_match_count >= min_matches and    # Lower threshold: 2 matches
            confidence >= 0.08                    # Lower confidence: 8%
        )
        
        return is_match, confidence, best_offset, match_details[best_offset]

def detect_audio_with_fingerprint(clip_path, main_audio_path, verbose=False):
    """
    Fingerprint detection - EXACT SAME as final_working.ipynb
    """
    fingerprinter = AudioFingerprinter()
    
    if verbose:
        print("Loading audio files for fingerprinting...")
    
    # Load audio files
    clip_audio = fingerprinter.load_audio(clip_path)
    main_audio = fingerprinter.load_audio(main_audio_path)
    
    if clip_audio is None or main_audio is None:
        return False, 0
    
    if verbose:
        print(f"Clip duration: {len(clip_audio)/fingerprinter.sample_rate:.2f}s")
        print(f"Main audio duration: {len(main_audio)/fingerprinter.sample_rate:.2f}s")
    
    # Check if clip is too long
    if len(clip_audio) >= len(main_audio):
        if verbose:
            print("Clip is longer than main audio")
        return False, "exceeds_video_length"
    
    if verbose:
        print("Generating fingerprints...")
    
    # Generate fingerprints
    clip_fingerprints = fingerprinter.generate_fingerprint(clip_audio)
    main_fingerprints = fingerprinter.generate_fingerprint(main_audio)
    
    if verbose:
        print(f"Clip fingerprint hashes: {len(clip_fingerprints)}")
        print(f"Main audio fingerprint hashes: {len(main_fingerprints)}")
    
    if len(clip_fingerprints) == 0:
        if verbose:
            print("Could not generate fingerprints for clip")
        return False, 0
    
    if verbose:
        print("Matching fingerprints...")
    
    # Match fingerprints with sensitive thresholds
    is_match, confidence, best_offset, match_details = fingerprinter.match_fingerprints(
        clip_fingerprints, main_fingerprints, min_matches=2
    )
    
    if verbose:
        print("\\n" + "="*60)
        print("AUDIO FINGERPRINT DETECTION RESULTS")
        print("="*60)
        print(f"Audio present: {'YES' if is_match else '❌ NO'}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Matching hashes: {len(match_details)}")
        
        if is_match:
            # Convert offset to time
            time_offset = best_offset * fingerprinter.hop_length / fingerprinter.sample_rate
            print(f"Found at time offset: {time_offset:.2f} seconds")
            print(f"Match strength: {len(match_details)} hash matches")
        else:
            print("No significant fingerprint matches found")
            if len(match_details) > 0:
                print(f"   Best attempt: {len(match_details)} matches (needed ≥2)")
                print(f"   Best confidence: {confidence:.4f} (needed ≥0.08)")
    
    return is_match, confidence

def get_audio_duration(file_path):
    """Get audio file duration in seconds"""
    try:
        audio, sr = librosa.load(file_path, sr=None)
        return len(audio) / sr
    except:
        return 0

def get_video_duration(video_path):
    """Get video duration in seconds using FFprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-show_entries', 
            'format=duration', '-of', 'csv=p=0', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return float(result.stdout.strip())
        return 0
    except:
        return 0

def analyze_student_video(video_file, assets_folder):
    """
    Main function to analyze student video against audio assets
    Returns: List of detected audio asset names
    """
    # Check if files exist
    if not os.path.exists(video_file):
        return []
    
    if not os.path.exists(assets_folder):
        return []
    
    # Get video duration
    video_duration = get_video_duration(video_file)
    
    # Use original video file directly (same as notebook)
    main_audio_file = video_file
    
    # Find all audio assets
    audio_extensions = ["*.wav", "*.mp3", "*.m4a", "*.flac", "*.aac"]
    asset_files = []
    for ext in audio_extensions:
        asset_files.extend(glob.glob(os.path.join(assets_folder, ext)))
        # Also check subdirectories
        asset_files.extend(glob.glob(os.path.join(assets_folder, "**", ext), recursive=True))
    
    # Remove duplicates and sort
    asset_files = sorted(list(set(asset_files)))
    
    if not asset_files:
        return []
    
    # Test each asset
    detected_assets = []
    
    for asset_path in asset_files:
        asset_name = os.path.basename(asset_path)
        asset_duration = get_audio_duration(asset_path)
        
        # Check if asset is longer than video
        if asset_duration > video_duration:
            continue
        
        # Test with both detection methods
        try:
            # MFCC Detection
            mfcc_result, mfcc_conf = is_audio_present(asset_path, main_audio_file, verbose=False)
            
            # Fingerprint Detection  
            fp_result, fp_conf = detect_audio_with_fingerprint(asset_path, main_audio_file, verbose=False)
            
            # Combined result (OR logic - same as notebook)
            overall_detected = mfcc_result or fp_result
            
            if overall_detected:
                detected_assets.append(asset_name)
                
        except Exception as e:
            continue
    
    return detected_assets

class AudioAssetMatcher:
    """
    Audio Asset Matcher Class for integration with the evaluation system
    """
    
    def __init__(self):
        """Initialize the audio asset matcher"""
        pass
    
    def detect_audio_assets(self, student_video_path, assets_folder_path):
        """
        Detect which audio assets are present in the student video
        
        Args:
            student_video_path (str): Path to student video file
            assets_folder_path (str): Path to assets folder containing audio files
            
        Returns:
            tuple: (detected_assets_list, total_assets_count, detected_count, asset_score)
        """
        try:
            # Use the exact same function from your code
            detected_assets = analyze_student_video(student_video_path, assets_folder_path)
            
            # Count total audio assets available
            if not os.path.exists(assets_folder_path):
                return [], 0, 0, 0.0
            
            audio_extensions = ["*.wav", "*.mp3", "*.m4a", "*.flac", "*.aac"]
            total_asset_files = []
            for ext in audio_extensions:
                total_asset_files.extend(glob.glob(os.path.join(assets_folder_path, ext)))
                # Also check subdirectories
                total_asset_files.extend(glob.glob(os.path.join(assets_folder_path, "**", ext), recursive=True))
            
            # Remove duplicates
            total_asset_files = list(set(total_asset_files))
            total_assets_count = len(total_asset_files)
            detected_count = len(detected_assets)
            
            # Calculate audio asset score
            if total_assets_count == 0:
                asset_score = 0.0  # No assets to match
            else:
                asset_score = detected_count / total_assets_count
            
            return detected_assets, total_assets_count, detected_count, asset_score
            
        except Exception as e:
            print(f"Error in audio asset detection: {e}")
            return [], 0, 0, 0.0
    
    def combine_audio_scores(self, current_audio_score, audio_asset_score, total_audio_assets):
        """
        Combine current audio score with audio asset score according to the specified logic
        
        Args:
            current_audio_score (float): Current audio score (0-1)
            audio_asset_score (float): Audio asset matching score (0-1)
            total_audio_assets (int): Total number of audio assets available
            
        Returns:
            float: Combined audio score
        """
        if total_audio_assets == 0:
            # No audio assets exist, give full weight to current audio score
            return current_audio_score * 1.0
        else:
            # Combine with 50-50 weight
            return (current_audio_score * 0.75) + (audio_asset_score * 0.25)
