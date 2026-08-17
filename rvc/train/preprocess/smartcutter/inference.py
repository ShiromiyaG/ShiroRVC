import os
import sys
import torch
import torchaudio
import torchaudio.functional as F_audio
import soundfile as sf
import glob
import numpy as np
import math
import gc

from rvc.train.preprocess.smartcutter.model_v5 import CGA_ResUNet
from rvc.train.preprocess.smartcutter.model_v3 import DSCA_ResUNet_v3

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------

#  INFER AND PROCESSING CONFIG
MODEL_VERSION = "v3"
FORCE_CPU = False                                       #   By default runs with GPU Acceleration ( CUDA )
MASK_MODE = "Soft"                                      #   Available:  "Soft", "Hard", "PowerMean" and "Hybrid"
DEBUG_MASK_PRED = False                                 #   Set to True if you need to debug / predict the model's prediction on your samples.
SAVE_EXTENSION = "wave_16"                              #   Available:  "flac", "wave_16" and "wave_32float"

#  SMART CUTTER CONFIG ( Safe defaults. )
SILENCE_TARGET_DURATION = 0.100                         #   The target duration for silence gaps (e.g. 500ms gap / silence --> 100ms)
MIN_SEGMENT_DURATION_MS = 100                           #   Minimum length for detected spots to count as viable for cutting ( 100ms, safe default. )

#  PREDICTION STABILIZATION
STABILITY_NOISE = False                                 #   Injects subtle noise into pure silence to stabilize the model
STABILITY_DB_LEVEL = -75.0                              #   The dB level of the injected noise  ( -80 is minimum;  Model's limitation. )
STABILITY_FADE_MS = 1                                   #   Fade duration (ms) for the injected noise edges to be softer
ENABLE_BRIDGING = True                                  #   Filling of the mask/prediction gaps - Only use when and if you debug the mask output and notice gaps.

#  PATHS
IN_DIR = "infer_input"
OUT_DIR = "infer_output"
CKPT_DIR = "ckpts"

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------



# Estabilished safe params, do not tweak these unless necessary and you know what you're doing.
SEARCH_WINDOW_MS = 25
FADE_DURATION_MS = 10
CUTTING_PROBABILITY = 0.5
SAFETY_BUFFER_MS = 5
SEGMENT_LEN = 8.0



def get_cosine_fade(length, device):
    # Generates a raised cosine curve (half-Hanning).
    # It's mathematically smoother than a linear fade, protecting spectral integrity.
    t = torch.linspace(0, math.pi, length, device=device)
    fade_curve = 0.5 * (1 - torch.cos(t))
    return fade_curve

def apply_fade(waveform, fade_samples, mode="both"):
    # Applies the cosine fade
    if waveform.shape[1] < fade_samples * 2:
        return waveform

    fade_curve = get_cosine_fade(fade_samples, waveform.device)

    if mode == "in" or mode == "both":
        # Fade In: curve goes 0 -> 1
        waveform[:, :fade_samples] *= fade_curve

    if mode == "out" or mode == "both":
        # Fade Out: flip the curve so it goes 1 -> 0
        waveform[:, -fade_samples:] *= fade_curve.flip(0)

    return waveform

def inject_stability_noise(wav, sr, device):
    """
    Injects steady, neutral colored noise
    """
    noise_amp = 10 ** (STABILITY_DB_LEVEL / 20.0)

    silence_mask = (wav.squeeze(0) == 0.0).float()
    diff = torch.diff(silence_mask, prepend=torch.tensor([0.0], device=device), append=torch.tensor([0.0], device=device))
    starts = torch.where(diff == 1)[0]
    ends = torch.where(diff == -1)[0]

    if len(starts) == 0:
        return wav

    raw_noise = torch.randn_like(wav)

    alpha = 0.85
    neutral_noise = torchaudio.functional.lfilter(
        raw_noise, 
        torch.tensor([1.0, 0.0], device=device), 
        torch.tensor([1.0, -alpha], device=device)
    )
    neutral_noise *= noise_amp
    fade_samples = int(sr * (STABILITY_FADE_MS / 1000.0))

    for start, end in zip(starts, ends):
        length = end - start
        if length <= 0: continue

        noise_chunk = neutral_noise[:, start:end].clone()

        if length > fade_samples * 2:
            noise_chunk = apply_fade(noise_chunk, fade_samples, mode="both")
        else:
            window = torch.hann_window(length, device=device)
            noise_chunk *= window

        wav[:, start:end] = noise_chunk

    return wav

def find_cuts(mask, waveform, sr):
    # Heavy lifter for phase-aware cutting:
    # 1. Finds rough cut points based on the prediction mask.
    # 2. FILTERS segments < 100ms
    # 3. Scans the actual audio for Zero Crossings (where signal crosses 0).
    # 4. Snaps the rough cuts to the nearest Zero Crossing to prevent clicking/popping.

    if mask.device.type != 'cpu': mask = mask.cpu()
    if waveform.device.type != 'cpu': waveform = waveform.cpu()

    mask_binary = (mask > CUTTING_PROBABILITY).float()

    # Identify edges of silence: 1 is start of silence, -1 is end.
    diff = torch.diff(mask_binary, prepend=torch.tensor([0]), append=torch.tensor([0]))

    rough_starts = torch.where(diff == 1)[0]
    rough_ends = torch.where(diff == -1)[0]

    if len(rough_starts) == 0:
        return [], []

    # Convert ms to samples
    min_samples = int(sr * (MIN_SEGMENT_DURATION_MS / 1000.0))

    # Calculate durations for all found segments
    durations = rough_ends - rough_starts

    # Keep only indices where duration >= min_samples
    valid_indices = torch.where(durations >= min_samples)[0]

    if len(valid_indices) == 0:
        return [], []

    # Update starts and ends to only include valid segments
    rough_starts = rough_starts[valid_indices]
    rough_ends = rough_ends[valid_indices]

    buffer_samples = int(sr * (SAFETY_BUFFER_MS / 1000.0))
    
    rough_starts = rough_starts + buffer_samples
    rough_ends = rough_ends - buffer_samples

    # Ensure we didn't invert the segment (start > end) after shrinking
    # This theoretically shouldn't happen if Min > 2*Buffer (100ms > 10ms), but better safe than sorry.
    valid_shrink = rough_ends > rough_starts
    rough_starts = rough_starts[valid_shrink]
    rough_ends = rough_ends[valid_shrink]
    # Flatten to mono to find crossings.
    wav_mono = waveform.mean(dim=0) 

    # Calculate differences in sign to find exact crossing indices.
    zero_crossings = torch.diff(torch.sign(wav_mono))

    # We want indices where the sign actually changed.
    valid_zc_indices = torch.where(zero_crossings != 0)[0]

    # If the audio is pure silence or DC offset, fallback to rough cuts.
    if len(valid_zc_indices) == 0:
        return rough_starts, rough_ends

    # Helper to find the index in 'candidates' closest to 'targets'.
    def snap_to_nearest(targets, candidates):
        # Find insertion points.
        idx = torch.searchsorted(candidates, targets)

        # Keep within array bounds.
        idx = torch.clamp(idx, 0, len(candidates) - 1)
        prev_idx = torch.clamp(idx - 1, 0, len(candidates) - 1)

        val_at_idx = candidates[idx]
        val_at_prev = candidates[prev_idx]

        # Check if the previous neighbor was actually closer.
        dist_idx = torch.abs(targets - candidates[idx])
        dist_prev = torch.abs(targets - candidates[prev_idx])

        return torch.where(dist_prev < dist_idx, candidates[prev_idx], candidates[idx])

    search_win_samples = int(sr * (SEARCH_WINDOW_MS / 1000))

    # Snap rough cuts to the nearest safe zero-crossing.
    safe_starts = snap_to_nearest(rough_starts, valid_zc_indices)
    safe_ends = snap_to_nearest(rough_ends, valid_zc_indices)

    # If the nearest ZC is too far away, just use the rough cut.
    start_diff = torch.abs(safe_starts - rough_starts)
    safe_starts = torch.where(start_diff < search_win_samples, safe_starts, rough_starts)

    end_diff = torch.abs(safe_ends - rough_ends)
    safe_ends = torch.where(end_diff < search_win_samples, safe_ends, rough_ends)

    return safe_starts, safe_ends

def SmartCutter(waveform, mask, sr=48000):
    waveform = waveform.cpu()
    mask = mask.cpu()

    if ENABLE_BRIDGING:
        # Gap bridge operation
        bridge_frames = 5 # At 100fps, 50ms is ~5 frames
        mask = mask.view(1, 1, -1)

        mask = torch.nn.functional.max_pool1d(mask, bridge_frames, 1, bridge_frames//2) # Dilation
        mask = -torch.nn.functional.max_pool1d(-mask, bridge_frames, 1, bridge_frames//2) # Erosion

    # Interpolate the low-res mask up to the full audio resolution.
    if mask.dim() == 1: mask = mask.view(1, 1, -1)
    elif mask.dim() == 2: mask = mask.unsqueeze(1)

    target_size = waveform.shape[1]

    mask_full = torch.nn.functional.interpolate(
        mask, size=target_size, mode='linear', align_corners=True
    ).squeeze()

    # Calculate where to cut.
    starts, ends = find_cuts(mask_full, waveform, sr)

    # If mask is empty, return silence.
    if len(starts) == 0:
        return torch.zeros_like(waveform), mask_full

    target_silence_samples = int(sr * SILENCE_TARGET_DURATION)
    fade_samples = int(sr * (FADE_DURATION_MS / 1000))

    pieces = []
    last_valid_idx = 0

    # Pre-allocate silence
    silence_tensor = torch.zeros((waveform.shape[0], target_silence_samples))

    start_list = starts.tolist()
    end_list = ends.tolist()

    for start_idx, end_idx in zip(start_list, end_list):
        start_idx = int(start_idx)
        end_idx = int(end_idx)

        # Only process if there's actual data to cut.
        if start_idx > last_valid_idx:
            speech_chunk = waveform[:, last_valid_idx:start_idx].clone()

            # Apply fades to the edges of the chunk to prevent clicks.
            if last_valid_idx > 0:
                speech_chunk = apply_fade(speech_chunk, fade_samples, mode="in")

            speech_chunk = apply_fade(speech_chunk, fade_samples, mode="out")
            pieces.append(speech_chunk)

        # Insert clean silence between chunks.
        pieces.append(silence_tensor)
        last_valid_idx = end_idx

    # Handle any remaining audio at the end.
    if last_valid_idx < target_size:
        tail_chunk = waveform[:, last_valid_idx:].clone()
        if last_valid_idx > 0:
            tail_chunk = apply_fade(tail_chunk, fade_samples, mode="in")
        pieces.append(tail_chunk)

    # Merge everything back into one tensor.
    return torch.cat(pieces, dim=1), mask_full

def process_grid_aligned(model, transform, waveform, sr, hop_length, device, static_input_buffer):
    # Implements Weighted Overlap-Add (WOLA) inference.
    # Processes audio in overlapping chunks and averages the results.
    total_samples = waveform.shape[1]

    CHUNK_SEC = SEGMENT_LEN
    OVERLAP_SEC = CHUNK_SEC / 2

    chunk_samples = int(CHUNK_SEC * sr)
    overlap_samples = int(OVERLAP_SEC * sr)
    stride_samples = chunk_samples - overlap_samples

    # Dummy pass to get dimensions
    dummy_input = torch.zeros(1, chunk_samples, device=device)
    dummy_mel = transform(dummy_input)
    frames_per_chunk = dummy_mel.shape[-1]

    # Total framess estimation for CPU buffer allocation
    total_frames = int(math.ceil(total_samples / hop_length)) + 100 # ample buffer

    print(f"    -> WOLA chunking: Chunk={chunk_samples}, Overlap={overlap_samples}, Total Frames={total_frames}")

    # Accumulators for the final mask and the window weights.
    mask_accumulator = torch.zeros((1, total_frames), dtype=torch.float32, device='cpu')
    weight_accumulator = torch.zeros((1, total_frames), dtype=torch.float32, device='cpu')

    # Hanning window ensures the center of the prediction counts more than the edges.
    window = torch.hann_window(frames_per_chunk, device=device).view(1, -1)
    # Moving window to CPU for accumulation later
    window_cpu = window.cpu()

    current_sample = 0

    with torch.no_grad():
        while current_sample < total_samples:
            start = current_sample
            end = start + chunk_samples

            chunk_wav = waveform[:, start:end]

            # Pad
            original_len = chunk_wav.shape[1]
            if original_len < chunk_samples:
                pad_amt = chunk_samples - original_len
                chunk_wav = torch.nn.functional.pad(chunk_wav, (0, pad_amt))

            # Move a chunk to GPU
            chunk_wav = chunk_wav.to(device)

            # Inference
            raw_mask = _run_inference(model, transform, chunk_wav, device, static_input_buffer)

            if raw_mask.dim() == 3: raw_mask = raw_mask.squeeze(1)

            # Map to frames
            start_frame = int(round(start / hop_length))

            # Ensure we don't go out of bounds
            if start_frame + frames_per_chunk > mask_accumulator.shape[1]:
                # Expand CPU buffer dynamically if needed
                extra = (start_frame + frames_per_chunk) - mask_accumulator.shape[1]
                mask_accumulator = torch.nn.functional.pad(mask_accumulator, (0, extra))
                weight_accumulator = torch.nn.functional.pad(weight_accumulator, (0, extra))

            # Accumulate on CPU
            current_pred_cpu = raw_mask.cpu() # Move pred to CPU

            # Add weighted prediction to accumulator.
            mask_accumulator[:, start_frame : start_frame + frames_per_chunk] += (current_pred_cpu * window_cpu)
            weight_accumulator[:, start_frame : start_frame + frames_per_chunk] += window_cpu

            current_sample += stride_samples

    # Normalize by weights to get the final average.
    weight_accumulator[weight_accumulator < 1e-6] = 1.0
    final_mask = mask_accumulator / weight_accumulator

    # Trim to actual size based on input waveform
    actual_frames = int(total_samples / hop_length)
    final_mask = final_mask[:, :actual_frames]

    return final_mask

def _run_inference(model, mel_transform, wav_chunk, device, input_buffer):
    # Standard forward pass: Waveform -> Mel -> Delta -> Model -> Mask.
    mel = mel_transform(wav_chunk).squeeze(0)
    mel = torchaudio.transforms.AmplitudeToDB()(mel)

    # Normalize dB to 0-1 range.
    min_db, max_db = -80.0, 0.0
    mel = torch.clamp(mel, min=min_db, max=max_db)
    mel = (mel - min_db) / (max_db - min_db)

    # Compute deltas for extra temporal context.
    delta = F_audio.compute_deltas(mel.unsqueeze(0)).squeeze(0)

    # buffer pushing
    current_frames = mel.shape[-1]
    input_buffer[0, 0, :, :current_frames].copy_(mel)
    input_buffer[0, 1, :, :current_frames].copy_(delta)

    # Model Inference using the buffer slice
    mask_2d = model(input_buffer[:, :, :, :current_frames])

    # Collapse 2D output (freq/time) to 1D (time) based on strategy.
    if MASK_MODE == "Soft":
        mask_pred = torch.mean(mask_2d, dim=2)
    elif MASK_MODE == "Hybrid":
        soft_mask = torch.mean(mask_2d, dim=2)
        hard_mask = torch.max(mask_2d, dim=2)[0]
        mask_pred = (0.7 * soft_mask) + (0.3 * hard_mask)
    elif MASK_MODE == "PowerMean":
        mask_pred = torch.sqrt(torch.mean(mask_2d**2, dim=2))
    elif MASK_MODE == "Hard":
        mask_pred = torch.max(mask_2d, dim=2)[0]
    else:
        print(f"MASK_MODE: {MASK_MODE} is unsupported. Exiting.")
        sys.exit(1)

    return mask_pred

def processing():
    # Device setup
    if FORCE_CPU:
        device = torch.device("cpu")
        print("FORCE_CPU is True. Using CPU.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = False # For consistency we disable it.
        print(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.")

    os.makedirs(IN_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    files = glob.glob(os.path.join(IN_DIR, "*.wav")) + glob.glob(os.path.join(IN_DIR, "*.flac"))
    print(f"Found {len(files)} files.")

    loaded_models = {}

    # Loop
    for f_path in files:
        try:
            fname = os.path.basename(f_path)
            print(f"Processing: {fname}...")

            # Audio loading
            wav, sr = torchaudio.load(f_path)

            # If audio is stereo, we pick the one with the lowest DC offset
            if wav.shape[0] > 1:
                # Calculate absolute mean (DC offset) for each channel
                dc_offsets = torch.abs(wav.mean(dim=1))

                # Find the index of the channel with the minimum offset
                best_ch_idx = torch.argmin(dc_offsets)

                # Select that channel and keep dimensions as [1, Time]
                wav = wav[best_ch_idx].unsqueeze(0)
                print(f"    -> Converted Stereo to Mono (Selected Ch {best_ch_idx}, DC: {dc_offsets[best_ch_idx]:.6f})")

            wav_for_inference = wav.clone()

            # Safety norm on input
            input_peak = torch.abs(wav_for_inference).max()
            if input_peak > 0:
                target_peak = 0.9 
                wav_for_inference = wav_for_inference * (target_peak / input_peak)

            if STABILITY_NOISE:
                wav_for_inference = inject_stability_noise(wav_for_inference, sr, wav.device)

            # Dynamic model loading based on Sample Rate.
            current_hop = sr // 100
            if sr not in loaded_models:

                # Release previous model if switching SR
                if len(loaded_models) > 0:
                    print("Unloading previous model to free VRAM...")
                    loaded_models.clear()
                    gc.collect()
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

                model_path = os.path.join(CKPT_DIR, f"{MODEL_VERSION}_model_{sr}.pth")
                if not os.path.exists(model_path):
                    print(f"Skipping {fname}: No {MODEL_VERSION} model for {sr}Hz")
                    continue

                print(f"Loading {sr}Hz {MODEL_VERSION} model ...")

                if MODEL_VERSION == "v3":
                    model = DSCA_ResUNet_v3(n_channels=2, n_classes=1).to(device)   # v3
                elif MODEL_VERSION == "v5":
                    model = CGA_ResUNet(n_channels=2, n_classes=1).to(device)       # v5
                else:
                    print(f"'{MODEL_VERSION}' is not a valid model version choice. Exiting.")
                    sys.exit(1)

                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()

                # mel transform config
                if sr in [48000, 40000]:
                    N_FFT = 2048
                    N_MELS = 160
                else:
                    N_FFT = 1024 # for 32khz model variant.
                    N_MELS = 128

                mel_transform = torchaudio.transforms.MelSpectrogram(
                    sample_rate=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=current_hop
                ).to(device)

                # we're pre-allocating a static buffer for the model input
                # Shape: [Batch, Channels, Mel_Bins, Frames_per_60s_chunk]
                # Channels = 2 (Mel + Delta)
                dummy_frames = int(math.ceil((SEGMENT_LEN * sr) / current_hop)) + 5
                static_buffer = torch.zeros((1, 2, N_MELS, dummy_frames), device=device)

                loaded_models[sr] = (model, mel_transform, static_buffer)

            curr_model, curr_mel_transform, curr_buffer = loaded_models[sr]

            # Inference (GPU-accelerated, CPU accumulation)
            mel_mask = process_grid_aligned(curr_model, curr_mel_transform, wav_for_inference, sr, current_hop, device, curr_buffer)

            if device.type == 'cuda':
                torch.cuda.empty_cache()

            # SmartCutter on CPU.
            cleaned, binary_mask = SmartCutter(wav, mel_mask, sr=sr)

            # Debug section
            if DEBUG_MASK_PRED:
                if binary_mask.dim() > 2: binary_mask = binary_mask.squeeze(0)

                binary_mask_interpolated = torch.nn.functional.interpolate(
                    binary_mask.view(1,1,-1), size=wav.shape[1], mode='nearest'
                ).squeeze()

                debug_noise = torch.randn_like(wav) * 0.09
                debug_wav = wav + (debug_noise * (binary_mask_interpolated > CUTTING_PROBABILITY).float())
                torchaudio.save(os.path.join(OUT_DIR, "debug_" + os.path.basename(f_path)), debug_wav, sr)


            # Volume Normalization
            peak = torch.abs(cleaned).max()
            if peak >= 0.95:
                scale_factor = 0.95 / peak.item()
                cleaned = cleaned * scale_factor

            # Output path construction
            file_stem = os.path.splitext(fname)[0]
            if SAVE_EXTENSION == "flac":
                out_path = os.path.join(OUT_DIR, file_stem + ".flac")
            elif "wave" in SAVE_EXTENSION:
                out_path = os.path.join(OUT_DIR, file_stem + ".wav")

            # Saving
            if SAVE_EXTENSION == "flac":
                torchaudio.save(out_path, cleaned, sr, format="flac", backend="soundfile")
            elif SAVE_EXTENSION == "wave_16":
                torchaudio.save(out_path, cleaned, sr, encoding="PCM_S", bits_per_sample=16)
            elif SAVE_EXTENSION == "wave_32float":
                torchaudio.save(out_path, cleaned, sr, encoding="PCM_F", bits_per_sample=32)
            else:
                print(f"Specified saving extension: '{SAVE_EXTENSION}' is unsupported. Exiting.")
                sys.exit(1)

            print(f"Saved: {out_path}")

            # Cleanup

            # inputs and outputs
            if 'wav' in locals(): del wav
            if 'cleaned' in locals(): del cleaned
            if 'wav_for_inference' in locals(): del wav_for_inference

            # masks and intermediate tensors
            if 'binary_mask' in locals(): del binary_mask
            if 'mel_mask' in locals(): del mel_mask

            # Delete debug pieces
            if 'debug_wav' in locals(): del debug_wav
            if 'debug_noise' in locals(): del debug_noise
            if 'binary_mask_interpolated' in locals(): del binary_mask_interpolated

            # Current buffer ( GPU )
            if 'curr_buffer' in locals(): curr_buffer.zero_()

            gc.collect()

            if device.type == 'cuda':
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing {f_path}: {e}")
            del e
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            continue


class SmartCutterInterface:
    def __init__(self, sr, ckpt_dir, device="cuda"):
        self.sr = sr
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_version = MODEL_VERSION
        self.ckpt_dir = ckpt_dir

        self.model = None
        self.mel_transform = None
        self.static_buffer = None
        self.loaded = False

    def load_model(self):
        if self.loaded: return

        print(f"[SmartCutter] Loading model on {self.device}...")

        # Load Model
        model_path = os.path.join(self.ckpt_dir, f"{self.model_version}_model_{self.sr}.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[SmartCutter] Model not found: {model_path}")

        if self.model_version == "v3":
            self.model = DSCA_ResUNet_v3(n_channels=2, n_classes=1).to(self.device)
        elif self.model_version == "v5":
            self.model = CGA_ResUNet(n_channels=2, n_classes=1).to(self.device)

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Config Mel
        curr_hop = self.sr // 100
        if self.sr in [48000, 40000]: n_fft, n_mels = 2048, 160
        else: n_fft, n_mels = 1024, 128

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr, n_mels=n_mels, n_fft=n_fft, hop_length=curr_hop
        ).to(self.device)

        # Pre-allocate buffer
        dummy_frames = int(math.ceil((SEGMENT_LEN * self.sr) / curr_hop)) + 5
        self.static_buffer = torch.zeros((1, 2, n_mels, dummy_frames), device=self.device)
        self.loaded = True

    def process_file(self, input_path, output_path):
        """
        Reads file, runs SmartCutter, saves to output_path.
        Returns: True if successful, False if failed.
        """
        if not self.loaded: self.load_model()

        try:
            # Load
            wav, load_sr = torchaudio.load(input_path)

            # Resample if needed (SmartCutter models are SR specific)
            if load_sr != self.sr:
                resampler = torchaudio.transforms.Resample(load_sr, self.sr).to(wav.device)
                wav = resampler(wav)

            # Mono check
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)

            wav_gpu = wav.to(self.device)

            # Norm input
            peak = torch.abs(wav_gpu).max()
            if peak > 0: wav_gpu = wav_gpu * (0.9 / peak)

            # Inference
            curr_hop = self.sr // 100
            mask = process_grid_aligned(
                self.model, self.mel_transform, wav_gpu, 
                self.sr, curr_hop, self.device, self.static_buffer
            )

            # Cutting
            cleaned, _ = SmartCutter(wav, mask, sr=self.sr)

            # Save
            torchaudio.save(output_path, cleaned, self.sr)
            return True

        except Exception as e:
            print(f"[SmartCutter] Error on {input_path}: {e}")
            return False

    def unload(self):
        # Free VRAM
        del self.model
        del self.mel_transform
        del self.static_buffer
        self.loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    processing()